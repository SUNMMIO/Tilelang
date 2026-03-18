#include "codegen_sunmmio.h"
#include "runtime/cuda/cuda_module.h"
#include "runtime/pack_args.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

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
  CodeGenTileLangSunMMIO cg;
  cg.Init();

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangSunMMIO: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_sunmmio_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return runtime::CUDAModuleCreate("ptx", "ptx", ExtractFuncInfo(mod), code);
}

ffi::Module BuildTileLangSunMMIOWithoutCompile(IRModule mod, Target target) {
  return BuildTileLangSunMMIO(mod, target);
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
