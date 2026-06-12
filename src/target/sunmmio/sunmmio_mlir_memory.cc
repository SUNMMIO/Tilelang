#include "sunmmio_mlir_memory.h"

#include "npuir/Dialect/SUVM/IR/Ops.h"
#include "npuir/Dialect/SUVM/IR/Types.h"
#include "tvm/runtime/logging.h"

namespace tvm {
namespace codegen {

SunmmioMlirMemory::SunmmioMlirMemory(SunmmioMlirContext &ctx)
    : ctx_(ctx), type_(ctx) {}

SunMMIOValue
SunmmioMlirMemory::Alloc(const std::string &result_name,
                         const SunMMIOType &memref_type,
                         const std::vector<SunMMIOValue> &dyn_extents,
                         const std::string &scope_name, DataType dtype) {
  if (memref_type.kind != SunMMIOType::Kind::kMemTensor) {
    LOG(FATAL) << "SunMMIO SUVM alloc expects memtensor type, but got kind "
               << static_cast<int>(memref_type.kind);
  }
  if (!dyn_extents.empty()) {
    LOG(FATAL) << "SunMMIO SUVM alloc does not support dynamic extents yet";
  }

  SunMMIOType updated_type = memref_type;
  if (updated_type.memory_scope.empty()) {
    updated_type.memory_scope = scope_name;
  }
  mlir::Type mapped_type = type_.MapType(updated_type);
  auto tensor_type = mlir::dyn_cast<mlir::suvm::MemTensorType>(mapped_type);
  if (!tensor_type) {
    LOG(FATAL) << "SunMMIO SUVM alloc expects suvm.memtensor result type";
  }

  mlir::suvm::AllocOp alloc = mlir::suvm::AllocOp::create(
      ctx_.builder, ctx_.builder.getUnknownLoc(), tensor_type);
  mlir::suvm::MemorySpace memory_space =
      tensor_type.getMemorySpace().getValue();
  if (memory_space == mlir::suvm::MemorySpace::asram ||
      memory_space == mlir::suvm::MemorySpace::wsram) {
    alloc->setAttr("suvm.ping_pong",
                   mlir::suvm::PingPongAttr::get(&ctx_.mlir_ctx,
                                                 mlir::suvm::PingPong::ping));
  }

  SunMMIOValue out{dtype, result_name, updated_type};
  ctx_.BindMLIRValue(result_name, alloc.getResult());
  return out;
}

SunMMIOValue SunmmioMlirMemory::Load(const std::string &result_name,
                                     const std::string &buffer_handle,
                                     const std::vector<SunMMIOValue> &indices,
                                     const SunMMIOType &memref_type,
                                     DataType dtype,
                                     const SunMMIOType &result_type) {
  (void)buffer_handle;
  (void)indices;
  (void)memref_type;
  ICHECK(ctx_.module)
      << "MLIR module must be initialized before lowering Sunmmio memory ops";
  mlir::TypedAttr value_attr = ctx_.builder.getIntegerAttr(
      mlir::Type::getFromOpaquePointer(
          ctx_.builder.getF32Type().getAsOpaquePointer()),
      0);
  auto fake_op = mlir::arith::ConstantOp::create(
      ctx_.builder, SunmmioMlirType(ctx_).MakeDebugLoc("fake_load"),
      value_attr);
  fake_op->setAttr("sunmmio.fake", ctx_.builder.getStringAttr("load"));
  mlir::Value load_value = fake_op.getResult();
  ctx_.BindMLIRValue(result_name, load_value);
  return SunMMIOValue{dtype, result_name, result_type};
}

void SunmmioMlirMemory::Store(const SunMMIOValue &value,
                              const std::string &buffer_handle,
                              const std::vector<SunMMIOValue> &indices,
                              const SunMMIOType &memref_type) {
  (void)value;
  (void)buffer_handle;
  (void)indices;
  (void)memref_type;
}

} // namespace codegen
} // namespace tvm
