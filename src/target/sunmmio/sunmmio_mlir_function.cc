#include "sunmmio_mlir_function.h"

namespace tvm {
namespace codegen {

SunmmioMlirFunction::SunmmioMlirFunction(SunmmioMlirContext& ctx) : ctx_(ctx) {}

void SunmmioMlirFunction::BeginModule() { ctx_.module_open = true; }

void SunmmioMlirFunction::EndModule() { ctx_.module_open = false; }

void SunmmioMlirFunction::BeginFunction(const std::string& name,
                                        const std::vector<BuilderArg>& args) {
  (void)args;
  ctx_.function_open = true;
  ctx_.current_function = name;
}

void SunmmioMlirFunction::EndFunction() {
  ctx_.function_open = false;
  ctx_.current_function.clear();
}

void SunmmioMlirFunction::EmitReturn() {}

void SunmmioMlirFunction::BeginFor(const std::string& iv,
                                   const SunMMIOValue& lb,
                                   const SunMMIOValue& ub,
                                   const SunMMIOValue& step) {
  (void)iv;
  (void)lb;
  (void)ub;
  (void)step;
  ctx_.insertion_point_stack.push_back("for");
}

void SunmmioMlirFunction::EndFor() {
  if (!ctx_.insertion_point_stack.empty()) {
    ctx_.insertion_point_stack.pop_back();
  }
}

void SunmmioMlirFunction::BeginIf(const SunMMIOValue& cond) {
  (void)cond;
  ctx_.insertion_point_stack.push_back("if.then");
}

void SunmmioMlirFunction::BeginElse() {
  if (!ctx_.insertion_point_stack.empty()) {
    ctx_.insertion_point_stack.back() = "if.else";
  }
}

void SunmmioMlirFunction::EndIf() {
  if (!ctx_.insertion_point_stack.empty()) {
    ctx_.insertion_point_stack.pop_back();
  }
}

void SunmmioMlirFunction::EmitAssert(const SunMMIOValue& cond,
                                     const std::string& msg_text) {
  (void)cond;
  (void)msg_text;
}

}  // namespace codegen
}  // namespace tvm
