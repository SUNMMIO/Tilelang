#include "codegen_sunmmio.h"
#include "target/source/codegen_source_base.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <tvm/ffi/reflection/registry.h>
#include <unordered_map>

namespace tvm {
namespace codegen {
namespace {

CodeGenTileLangSunMMIO::BuilderBackendKind ParseBackendKind(
    const ffi::String &backend) {
  std::string mode = static_cast<std::string>(backend);
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (mode == "text" || mode == "textdebug") {
    return CodeGenTileLangSunMMIO::BuilderBackendKind::kTextDebug;
  }
  if (mode == "suvm") {
    return CodeGenTileLangSunMMIO::BuilderBackendKind::kSuvm;
  }
  LOG(FATAL) << "Unknown sunmmio builder backend: " << mode
             << ". Expected one of: suvm, text.";
  TVM_FFI_UNREACHABLE();
}

} // namespace

static std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      DataType dtype = f->params[i].dtype();
      if (dtype.is_bool()) {
        dtype = DataType::Int(32);
      }
      info.arg_types.push_back(dtype);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tir::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        info.launch_param_tags.push_back(tag);
      }
    }
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    fmap[static_cast<std::string>(global_symbol.value())] = info;
  }
  return fmap;
}

ffi::Module BuildTileLangSunMMIO(IRModule mod, Target target) {
  LOG(FATAL) << "target.build.tilelang_sunmmio is not implemented yet. "
             << "Use target.build.tilelang_sunmmio_without_compile "
             << "or set enable_device_compile=False.";
  TVM_FFI_UNREACHABLE();
}

ffi::Module BuildTileLangSunMMIOWithoutCompile(IRModule mod, Target target,
                                               ffi::String backend) {
  CodeGenTileLangSunMMIO cg(ParseBackendKind(backend));
  cg.Init();

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenTileLangSunMMIO: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    // ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_sunmmio_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return codegen::DeviceSourceModuleCreate(code, "sunmmio",
                                           ExtractFuncInfo(mod), "sunmmio");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_sunmmio", BuildTileLangSunMMIO)
      .def("target.build.tilelang_sunmmio_without_compile",
           BuildTileLangSunMMIOWithoutCompile);
}

} // namespace codegen
} // namespace tvm
