#ifndef TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
#define TVM_TL_TARGET_CODEGEN_SUNMMIO_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

struct SunMMIOValue {
  DataType dtype;
  std::string value;
  std::string mlir_type;
};

struct BufferBinding {
  tir::Buffer buffer;
  std::string handle;
  std::string memref_type;
  std::string scope;
  bool is_external{false};
};

class CodeGenTileLangSunMMIO final : public tir::StmtVisitor,
                                      public tir::ExprFunctor<SunMMIOValue(const tvm::PrimExpr&)> {
public:
  CodeGenTileLangSunMMIO();
  ~CodeGenTileLangSunMMIO() noexcept override = default;

  void Init();
  void Clear();
  void AddFunction(const GlobalVar& gvar, const tir::PrimFunc& f);
  std::string Finish();

protected:
  void VisitStmt_(const tir::SeqStmtNode* op) override;
  void VisitStmt_(const tir::ForNode* op) override;
  void VisitStmt_(const tir::LetStmtNode* op) override;
  void VisitStmt_(const tir::AttrStmtNode* op) override;
  void VisitStmt_(const tir::IfThenElseNode* op) override;
  void VisitStmt_(const tir::WhileNode* op) override;
  void VisitStmt_(const tir::AllocateNode* op) override;
  void VisitStmt_(const tir::AllocateConstNode* op) override;
  void VisitStmt_(const tir::DeclBufferNode* op) override;
  void VisitStmt_(const tir::BufferStoreNode* op) override;
  void VisitStmt_(const tir::BufferRealizeNode* op) override;
  void VisitStmt_(const tir::AssertStmtNode* op) override;
  void VisitStmt_(const tir::EvaluateNode* op) override;
  void VisitStmt_(const tir::BlockNode* op) override;
  void VisitStmt_(const tir::BlockRealizeNode* op) override;
  void VisitStmtDefault_(const Object* op) override;

  SunMMIOValue VisitExpr_(const tir::VarNode* op) override;
  SunMMIOValue VisitExpr_(const tir::SizeVarNode* op) override;
  SunMMIOValue VisitExpr_(const tir::IntImmNode* op) override;
  SunMMIOValue VisitExpr_(const tir::FloatImmNode* op) override;
  SunMMIOValue VisitExpr_(const tir::StringImmNode* op) override;
  SunMMIOValue VisitExpr_(const tir::CastNode* op) override;
  SunMMIOValue VisitExpr_(const tir::CallNode* op) override;
  SunMMIOValue VisitExpr_(const tir::AddNode* op) override;
  SunMMIOValue VisitExpr_(const tir::SubNode* op) override;
  SunMMIOValue VisitExpr_(const tir::MulNode* op) override;
  SunMMIOValue VisitExpr_(const tir::DivNode* op) override;
  SunMMIOValue VisitExpr_(const tir::ModNode* op) override;
  SunMMIOValue VisitExpr_(const tir::FloorDivNode* op) override;
  SunMMIOValue VisitExpr_(const tir::FloorModNode* op) override;
  SunMMIOValue VisitExpr_(const tir::MinNode* op) override;
  SunMMIOValue VisitExpr_(const tir::MaxNode* op) override;
  SunMMIOValue VisitExpr_(const tir::EQNode* op) override;
  SunMMIOValue VisitExpr_(const tir::NENode* op) override;
  SunMMIOValue VisitExpr_(const tir::LTNode* op) override;
  SunMMIOValue VisitExpr_(const tir::LENode* op) override;
  SunMMIOValue VisitExpr_(const tir::GTNode* op) override;
  SunMMIOValue VisitExpr_(const tir::GENode* op) override;
  SunMMIOValue VisitExpr_(const tir::AndNode* op) override;
  SunMMIOValue VisitExpr_(const tir::OrNode* op) override;
  SunMMIOValue VisitExpr_(const tir::NotNode* op) override;
  SunMMIOValue VisitExpr_(const tir::SelectNode* op) override;
  SunMMIOValue VisitExpr_(const tir::BufferLoadNode* op) override;
  SunMMIOValue VisitExpr_(const tir::ProducerLoadNode* op) override;
  SunMMIOValue VisitExpr_(const tir::RampNode* op) override;
  SunMMIOValue VisitExpr_(const tir::BroadcastNode* op) override;
  SunMMIOValue VisitExpr_(const tir::ShuffleNode* op) override;
  SunMMIOValue VisitExpr_(const tir::LetNode* op) override;
  SunMMIOValue VisitExprDefault_(const Object* op) override;

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

  SunMMIOValue EvalExpr(const tvm::PrimExpr& expr);
  SunMMIOValue EmitBinary(const char* op_name, const tvm::PrimExpr& lhs,
                          const tvm::PrimExpr& rhs, tvm::DataType dtype);
  SunMMIOValue EmitCmp(const char* pred, const tvm::PrimExpr& lhs,
                       const tvm::PrimExpr& rhs);
  SunMMIOValue EmitCast(const SunMMIOValue& v, tvm::DataType target_dtype);
  SunMMIOValue EmitCall(const tir::CallNode* op);
  SunMMIOValue EmitLoad(const tir::Buffer& buffer,
                        const ffi::Array<PrimExpr>& indices);
  void EmitStore(const tir::Buffer& buffer, const ffi::Array<PrimExpr>& indices,
                 const SunMMIOValue& value);
  void EmitAlloc(const tir::Var& buffer_var, DataType dtype,
                 const ffi::Array<PrimExpr>& extents,
                 const std::string& scope_hint);
  void EmitFor(const tir::ForNode* op);
  void EmitIf(const tir::IfThenElseNode* op);

  std::string MapType(tvm::DataType dtype) const;
  std::string MapBufferType(const tir::Buffer& buffer) const;
  std::string MapStorageScope(const std::string& scope) const;
  std::string ClassifyParamKind(const tir::Var& param,
                                const tir::PrimFunc& f) const;
  std::string NewValueName();
  std::string NewLabel(const std::string& prefix);
  void EmitLine(const std::string& line);
  SunMMIOValue EmitConstIndex(int64_t v);
  SunMMIOValue EnsureIndex(const SunMMIOValue& v);
  SunMMIOValue EnsureType(const SunMMIOValue& v, const std::string& mlir_type,
                          DataType dtype);
  SunMMIOValue BindVar(const tir::Var& var, const SunMMIOValue& value);
  void RegisterBuffer(const tir::Buffer& buffer, bool is_external,
                      const std::string& handle_hint = "");
  const BufferBinding& LookupBuffer(const tir::Buffer& buffer) const;
  void EnterScope();
  void ExitScope();

  CallBucket ClassifyCall(const tir::CallNode* op) const;
  const char* CallBucketName(CallBucket bucket) const;
  [[noreturn]] void UnsupportedStmt(const Object* op,
                                    const std::string& detail = "") const;
  [[noreturn]] void UnsupportedExpr(const Object* op,
                                    const std::string& detail = "") const;

  std::ostringstream mlir_;
  bool initialized_{false};
  int ssa_counter_{0};
  int label_counter_{0};
  int indent_{0};
  bool module_open_{false};
  bool function_open_{false};

  std::unordered_map<const tir::VarNode*, SunMMIOValue> var_table_;
  std::unordered_map<const tir::BufferNode*, BufferBinding> buffer_registry_;
  std::vector<ScopedAttr> attr_stack_;

  std::vector<const tir::VarNode*> scoped_vars_;
  std::vector<const tir::BufferNode*> scoped_buffers_;
  std::vector<size_t> var_scope_markers_;
  std::vector<size_t> buffer_scope_markers_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
