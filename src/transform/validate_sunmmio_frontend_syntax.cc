/*!
 * \file validate_sunmmio_frontend_syntax.cc
 * \brief Reject unsupported user-facing syntax before Sunmmio lowering.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
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
    if (op->block->annotations.count(attr::kExplicitWhere)) {
      LOG(FATAL) << "Sunmmio frontend does not support T.where; use an "
                    "explicit if statement instead. Block '"
                 << op->block->name_hint << "' has predicate " << op->predicate;
    }
    StmtVisitor::VisitStmt_(op);
  }
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
