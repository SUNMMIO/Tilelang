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

struct ExprInfo {
  DataType dtype;
  std::string kind;
  std::string text;
  bool is_constant{false};
};

struct BufferInfo {
  std::string name;
  tir::Buffer buffer;
  DataType dtype;
  ffi::Array<PrimExpr> shape;
  std::string scope;
  bool is_external{false};
};

struct LoopInfo {
  tir::Var loop_var;
  ExprInfo min;
  ExprInfo extent;
  tir::ForKind kind;
};

struct AttrInfo {
  ffi::Any node;
  ffi::String key;
  ExprInfo value;
};

struct FunctionInfo {
  GlobalVar gvar;
  tir::PrimFunc prim_func;
  std::string name;
  std::vector<std::string> param_names;
  std::vector<std::string> param_kinds;
};

class CodeGenTileLangSunMMIO final : public tir::StmtVisitor,
                                      public tir::ExprFunctor<ExprInfo(const tvm::PrimExpr&)> {
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

  ExprInfo VisitExpr_(const tir::VarNode* op) override;
  ExprInfo VisitExpr_(const tir::SizeVarNode* op) override;
  ExprInfo VisitExpr_(const tir::IntImmNode* op) override;
  ExprInfo VisitExpr_(const tir::FloatImmNode* op) override;
  ExprInfo VisitExpr_(const tir::StringImmNode* op) override;
  ExprInfo VisitExpr_(const tir::CastNode* op) override;
  ExprInfo VisitExpr_(const tir::CallNode* op) override;
  ExprInfo VisitExpr_(const tir::AddNode* op) override;
  ExprInfo VisitExpr_(const tir::SubNode* op) override;
  ExprInfo VisitExpr_(const tir::MulNode* op) override;
  ExprInfo VisitExpr_(const tir::DivNode* op) override;
  ExprInfo VisitExpr_(const tir::ModNode* op) override;
  ExprInfo VisitExpr_(const tir::FloorDivNode* op) override;
  ExprInfo VisitExpr_(const tir::FloorModNode* op) override;
  ExprInfo VisitExpr_(const tir::MinNode* op) override;
  ExprInfo VisitExpr_(const tir::MaxNode* op) override;
  ExprInfo VisitExpr_(const tir::EQNode* op) override;
  ExprInfo VisitExpr_(const tir::NENode* op) override;
  ExprInfo VisitExpr_(const tir::LTNode* op) override;
  ExprInfo VisitExpr_(const tir::LENode* op) override;
  ExprInfo VisitExpr_(const tir::GTNode* op) override;
  ExprInfo VisitExpr_(const tir::GENode* op) override;
  ExprInfo VisitExpr_(const tir::AndNode* op) override;
  ExprInfo VisitExpr_(const tir::OrNode* op) override;
  ExprInfo VisitExpr_(const tir::NotNode* op) override;
  ExprInfo VisitExpr_(const tir::SelectNode* op) override;
  ExprInfo VisitExpr_(const tir::BufferLoadNode* op) override;
  ExprInfo VisitExpr_(const tir::ProducerLoadNode* op) override;
  ExprInfo VisitExpr_(const tir::RampNode* op) override;
  ExprInfo VisitExpr_(const tir::BroadcastNode* op) override;
  ExprInfo VisitExpr_(const tir::ShuffleNode* op) override;
  ExprInfo VisitExpr_(const tir::LetNode* op) override;
  ExprInfo VisitExprDefault_(const Object* op) override;

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

  ExprInfo EvalExpr(const tvm::PrimExpr& expr);
  ExprInfo MakeExprInfo(DataType dtype, const char* kind, std::string text,
                        bool is_constant = false) const;
  ExprInfo EmitBinaryExprInfo(const char* kind, const tvm::PrimExpr& lhs,
                              const tvm::PrimExpr& rhs, tvm::DataType dtype);
  std::string PrintType(tvm::DataType dtype) const;
  void EnterScope();
  void ExitScope();
  void BindVar(const tir::Var& var, const ExprInfo& info);
  void RegisterBuffer(const tir::Buffer& buffer, bool is_external);
  void RecordAttr(const AttrInfo& attr);
  void RecordLoop(const LoopInfo& loop);
  void RecordFunctionMetadata(const GlobalVar& gvar, const tir::PrimFunc& f);
  std::string ClassifyParamKind(const tir::Var& param,
                                const tir::PrimFunc& f) const;
  CallBucket ClassifyCall(const tir::CallNode* op) const;
  std::string ToString(CallBucket bucket) const;
  [[noreturn]] void UnsupportedStmt(const Object* op,
                                    const std::string& detail = "") const;
  [[noreturn]] void UnsupportedExpr(const Object* op,
                                    const std::string& detail = "") const;

  std::ostringstream summary_;
  bool initialized_{false};
  int expr_counter_{0};

  std::vector<FunctionInfo> functions_;
  FunctionInfo* current_function_{nullptr};

  std::unordered_map<const tir::VarNode*, ExprInfo> var_table_;
  std::unordered_map<const tir::BufferNode*, BufferInfo> buffer_registry_;
  std::vector<LoopInfo> loop_stack_;
  std::vector<AttrInfo> attr_stack_;
  std::vector<std::string> diagnostics_;
  std::vector<std::string> unsupported_reports_;

  std::vector<const tir::VarNode*> scoped_vars_;
  std::vector<const tir::BufferNode*> scoped_buffers_;
  std::vector<size_t> var_scope_markers_;
  std::vector<size_t> buffer_scope_markers_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
