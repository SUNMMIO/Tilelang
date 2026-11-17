#include "sunmmio_mlir_dist.h"

#include "mlir/IR/Verifier.h"
#include "npuir/Dialect/SUVM/IR/Ops.h"

namespace tvm {
namespace codegen {
namespace {

template <typename Op> void VerifyDistOp(Op op) {
  ICHECK(mlir::succeeded(mlir::verify(op.getOperation())))
      << "Invalid SunMMIO dist operation";
  if (auto arch = mlir::dyn_cast<mlir::suvm::ArchVerifierOpInterface>(
          op.getOperation())) {
    ICHECK(
        mlir::succeeded(arch.verifyWithDeviceArch(mlir::suvm::DeviceArch::a4e)))
        << "SunMMIO dist operation violates A4E constraints";
  }
}

} // namespace

SunMMIOValue SunmmioMlirDist::Signal(const std::string &name,
                                     const SunMMIOType &type) {
  auto signal_type =
      mlir::cast<mlir::suvm::DistSignalType>(type_.MapType(type));
  auto op =
      mlir::suvm::DistSignalOp::create(ctx_.builder, type_.Loc(), signal_type);
  VerifyDistOp(op);
  ctx_.BindMLIRValue(name, op.getSignal());
  return {DataType::Handle(), name, type};
}

SunMMIOValue SunmmioMlirDist::BatchBegin(const std::string &name) {
  SunMMIOType type;
  type.kind = SunMMIOType::Kind::kDistBatch;
  type.dtype = DataType::Handle();
  auto op = mlir::suvm::DistBatchBeginOp::create(ctx_.builder, type_.Loc(),
                                                 type_.MapType(type));
  // The complete module verifier checks consumption after submit is emitted.
  ctx_.BindMLIRValue(name, op.getBatch());
  return {DataType::Handle(), name, type};
}

void SunmmioMlirDist::BatchPut(const SunMMIOValue &batch,
                               const SunMMIOValue &peer,
                               const SunMMIOValue &src,
                               const SunMMIOValue &dst) {
  auto op = mlir::suvm::DistBatchPutOp::create(
      ctx_.builder, type_.Loc(), ctx_.LookupValue(batch, "dist batch"),
      ctx_.LookupValue(src, "dist source"),
      ctx_.LookupValue(dst, "dist destination"),
      ctx_.LookupValue(peer, "dist peer"));
  VerifyDistOp(op);
}

void SunmmioMlirDist::BatchSignal(const SunMMIOValue &batch,
                                  const SunMMIOValue &peer,
                                  const SunMMIOValue &signal,
                                  const SunMMIOValue &index,
                                  const std::optional<SunMMIOValue> &value) {
  auto op = mlir::suvm::DistBatchSignalOp::create(
      ctx_.builder, type_.Loc(), ctx_.LookupValue(batch, "dist batch"),
      ctx_.LookupValue(signal, "dist signal"),
      ctx_.LookupValue(index, "dist member"),
      value ? ctx_.LookupValue(*value, "dist generation") : mlir::Value(),
      ctx_.LookupValue(peer, "dist peer"));
  VerifyDistOp(op);
}

void SunmmioMlirDist::BatchSubmit(const SunMMIOValue &batch) {
  auto op = mlir::suvm::DistBatchSubmitOp::create(
      ctx_.builder, type_.Loc(), ctx_.LookupValue(batch, "dist batch"));
  VerifyDistOp(op);
}

void SunmmioMlirDist::SignalWait(const SunMMIOValue &signal,
                                 const SunMMIOValue &index,
                                 const SunMMIOValue &expected) {
  mlir::Value value = ctx_.LookupValue(expected, "dist expected");
  auto op = mlir::suvm::DistSignalWaitOp::create(
      ctx_.builder, type_.Loc(), value.getType(),
      ctx_.LookupValue(signal, "dist signal"),
      ctx_.LookupValue(index, "dist member"), value, mlir::suvm::SignalCmp::ge);
  VerifyDistOp(op);
}

SunMMIOValue SunmmioMlirDist::SignalFetch(const std::string &name,
                                          const SunMMIOValue &signal,
                                          const SunMMIOValue &index,
                                          DataType dtype) {
  SunMMIOType result_type{SunMMIOType::Kind::kScalar, dtype, 1, {}};
  auto op = mlir::suvm::DistSignalFetchOp::create(
      ctx_.builder, type_.Loc(), type_.MapType(result_type),
      ctx_.LookupValue(signal, "dist signal"),
      ctx_.LookupValue(index, "dist member"));
  VerifyDistOp(op);
  ctx_.BindMLIRValue(name, op.getValue());
  return {dtype, name, result_type};
}

void SunmmioMlirDist::Flush() {
  mlir::suvm::DistFlushOp::create(ctx_.builder, type_.Loc());
}

} // namespace codegen
} // namespace tvm
