#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

tvm::transform::Pass LegalizeTilesLoop();
tvm::transform::Pass TilesLoop();

tvm::transform::Pass LowerTilesLoop() {
  auto pass_func = [](IRModule mod, const tvm::transform::PassContext &) {
    mod = LegalizeTilesLoop()(std::move(mod));
    mod = TilesLoop()(std::move(mod));
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tl.LowerTilesLoop",
                                          {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.LowerTilesLoop",
                                        LowerTilesLoop);
}

} // namespace tl
} // namespace tvm
