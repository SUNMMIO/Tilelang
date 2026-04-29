#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_TYPE_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_TYPE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

enum class BinaryOp {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kMin,
  kMax,
  kAnd,
  kOr,
  kXor
};

enum class ArithmeticFlavor { kFloat, kSignedInt, kUnsignedInt, kBool, kIndex };

enum class CompareOp { kEQ, kNE, kLT, kLE, kGT, kGE };

enum class CompareDomain { kFloat, kSignedInt, kUnsignedInt, kBool };

struct SunMMIOType {
  enum class Kind {
    kScalar,
    kIndex,
    kHandle,
    kVector,
    kMemRef,
    kMemTensor,
    kUnknown

  };

  Kind kind{Kind::kUnknown};
  DataType dtype{DataType::Void()};
  int lanes{1};
  std::vector<PrimExpr> shape;
  std::vector<int64_t> layout_hshape;
  std::vector<int64_t> layout_hstride;
  std::vector<uint8_t> layout_dim_levels;
  std::string memory_scope;
  int64_t byte_offset{0};
};

struct SunMMIOValue {
  DataType dtype;
  std::string value;
  SunMMIOType type;
};

struct BuilderArg {
  std::string name;
  SunMMIOType type;
};

struct BufferBinding {
  tir::Buffer buffer;
  std::string handle;
  SunMMIOType buffer_type;
  std::string scope;
  bool is_external{false};
};

struct SunmmioMlirContext;

class SunmmioMlirType {
public:
  explicit SunmmioMlirType(SunmmioMlirContext &ctx);

  mlir::Location Loc() const;
  mlir::Type MapElementType(DataType dtype) const;
  mlir::Type MapType(const SunMMIOType &type) const;
  mlir::Value EnsureI1(mlir::Value v);
  mlir::Value EnsureIndex(mlir::Value v);
  mlir::Value ResolveValueOrCreatePlaceholder(const SunMMIOValue &v,
                                              mlir::Type expected_type);

private:
  SunmmioMlirContext &ctx_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_TYPE_H_
