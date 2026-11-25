#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_DIST_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_DIST_H_

#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

class SunmmioMlirDist {
public:
  explicit SunmmioMlirDist(SunmmioMlirContext &ctx) : ctx_(ctx), type_(ctx) {}

  SunMMIOValue Signal(const std::string &name, const SunMMIOType &type);
  SunMMIOValue BatchBegin(const std::string &name);
  void BatchPut(const SunMMIOValue &batch, const SunMMIOValue &peer,
                const SunMMIOValue &src, const SunMMIOValue &dst);
  void BatchSignal(const SunMMIOValue &batch, const SunMMIOValue &peer,
                   const SunMMIOValue &signal, const SunMMIOValue &index,
                   const std::optional<SunMMIOValue> &value);
  void BatchSubmit(const SunMMIOValue &batch);
  void SignalWait(const SunMMIOValue &signal, const SunMMIOValue &index,
                  const SunMMIOValue &expected);
  SunMMIOValue SignalFetch(const std::string &name, const SunMMIOValue &signal,
                           const SunMMIOValue &index, DataType dtype);
  void Flush();

private:
  SunmmioMlirContext &ctx_;
  SunmmioMlirType type_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_DIST_H_
