/*!
 * \file validate_sunmmio_frontend_syntax.cc
 * \brief Reject unsupported user-facing syntax before Sunmmio lowering.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../target/utils.h"
#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

class ValidateSunmmioFrontendSyntaxPass : public StmtVisitor {
public:
  static PrimFunc Run(PrimFunc func) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsSunmmio(target.value())) {
      return func;
    }

    ValidateSunmmioFrontendSyntaxPass validator;
    validator.VisitStmt(func->body);
    return func;
  }

private:
  void VisitStmt_(const BlockRealizeNode *op) final {
    PrimExpr predicate = analyzer_.Simplify(op->predicate);
    if (op->block->annotations.count(attr::kExplicitWhere) ||
        !is_one(predicate)) {
      LOG(FATAL) << "Sunmmio frontend does not support T.where or non-trivial "
                    "block predicates; use an explicit if statement instead. "
                    "Block '"
                 << op->block->name_hint << "' has predicate " << predicate;
    }
    StmtVisitor::VisitStmt_(op);
  }

  arith::Analyzer analyzer_;
};

} // namespace

Pass ValidateSunmmioFrontendSyntax() {
  auto pass_func = [=](PrimFunc func, const IRModule &mod,
                       const PassContext &ctx) {
    return ValidateSunmmioFrontendSyntaxPass::Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ValidateSunmmioFrontendSyntax",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateSunmmioFrontendSyntax",
                        ValidateSunmmioFrontendSyntax);
}

} // namespace tl
} // namespace tvm
