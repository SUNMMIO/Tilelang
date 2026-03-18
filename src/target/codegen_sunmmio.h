#ifndef TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
#define TVM_TL_TARGET_CODEGEN_SUNMMIO_H_

#include <tvm/ir/global_var.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace codegen {

class CodeGenTileLangSunMMIO final : public tir::StmtVisitor,
                                      public tir::ExprFunctor<std::string(const tvm::PrimExpr&)> {
public:
  CodeGenTileLangSunMMIO();
  ~CodeGenTileLangSunMMIO() noexcept override = default;

  void Init();
  void Clear();
  void AddFunction(const GlobalVar& gvar, const tir::PrimFunc& f);
  std::string Finish();

protected:
  void VisitStmt_(const tir::ForNode* op) override;
  void VisitStmt_(const tir::BufferStoreNode* op) override;
  void VisitStmt_(const tir::AllocateNode* op) override;
  void VisitStmt_(const tir::AttrStmtNode* op) override;
  void VisitStmt_(const tir::IfThenElseNode* op) override;
  void VisitStmt_(const tir::EvaluateNode* op) override;

  std::string VisitExpr_(const tir::VarNode* op) override;
  std::string VisitExpr_(const tir::BufferLoadNode* op) override;
  std::string VisitExpr_(const tir::IntImmNode* op) override;
  std::string VisitExpr_(const tir::FloatImmNode* op) override;
  std::string VisitExpr_(const tir::AddNode* op) override;
  std::string VisitExpr_(const tir::SubNode* op) override;
  std::string VisitExpr_(const tir::MulNode* op) override;
  std::string VisitExpr_(const tir::DivNode* op) override;
  std::string VisitExpr_(const tir::CastNode* op) override;
  std::string VisitExpr_(const tir::CallNode* op) override;
  std::string VisitExprDefault_(const Object* op) override;

private:
  std::string NewValue();
  std::string EmitBinary(const char* op_name, const tvm::PrimExpr& lhs,
                         const tvm::PrimExpr& rhs, tvm::DataType dtype);
  std::string PrintType(tvm::DataType dtype) const;
  void EmitLine(const std::string& line);
  void EnterScope();
  void ExitScope();

  std::ostringstream stream_;
  int indent_{0};
  int value_counter_{0};
  bool initialized_{false};
  std::unordered_map<const tir::VarNode*, std::string> var_ids_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_SUNMMIO_H_
