#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_TILE_OP_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_TILE_OP_H_

#include "codegen_sunmmio.h"
#include "sunmmio_mlir_context.h"

namespace tvm {
namespace codegen {

class SunmmioMlirTileOp {
public:
  explicit SunmmioMlirTileOp(SunmmioMlirContext &ctx);

  SunMMIOValue GetPartitionedTileView(const std::string &result_name,
                                      const SunMMIOValue &memtensor,
                                      const std::vector<SunMMIOValue> &indices,
                                      const std::vector<int64_t> &tiled_dims,
                                      const SunMMIOType &view_type,
                                      DataType dtype);

  SunMMIOValue TileLoad(const std::string &result_name,
                        const SunMMIOValue &tile_view,
                        const SunMMIOType &tile_type,
                        const std::optional<SunMMIOValue> &mask,
                        const std::optional<SunMMIOValue> &maskedoff,
                        DataType dtype);

  SunMMIOValue TileFill(const std::string &result_name,
                        const SunMMIOValue &scalar,
                        const SunMMIOType &tile_type, DataType dtype);

  SunMMIOValue Cast(const std::string &result_name, const SunMMIOValue &v,
                    const SunMMIOType &dst_type, DataType dst_dtype);

  SunMMIOValue Binary(const std::string &result_name, BinaryOp op,
                      ArithmeticFlavor flavor, const SunMMIOValue &a,
                      const SunMMIOValue &b, const SunMMIOType &result_type,
                      DataType dtype);

  SunMMIOValue Unary(const std::string &result_name, TileUnaryOp op,
                     const SunMMIOValue &data, const SunMMIOType &result_type,
                     DataType dtype);

  SunMMIOValue Compare(const std::string &result_name, CompareOp op,
                       CompareDomain domain, const SunMMIOValue &a,
                       const SunMMIOValue &b, const SunMMIOType &operand_type);

  SunMMIOValue TileUnsqueeze(const std::string &result_name,
                             const SunMMIOValue &tile,
                             const SunMMIOType &tile_type, int64_t axis,
                             DataType dtype);

  SunMMIOValue TileBroadcast(const std::string &result_name,
                             const SunMMIOValue &tile,
                             const SunMMIOType &tile_type, DataType dtype);

  SunMMIOValue TileSlice(const std::string &result_name,
                         const SunMMIOValue &tile,
                         const std::vector<SunMMIOValue> &offsets,
                         const SunMMIOType &tile_type, DataType dtype);

  SunMMIOValue TileInsertSlice(const std::string &result_name,
                               const SunMMIOValue &base,
                               const SunMMIOValue &slice,
                               const std::vector<SunMMIOValue> &offsets,
                               const SunMMIOType &result_type, DataType dtype);

  SunMMIOValue TileRectMask(const std::string &result_name,
                            const SunMMIOValue &valid_rows,
                            const SunMMIOValue &valid_cols,
                            const SunMMIOType &tile_type);

  SunMMIOValue TileAxisMask(const std::string &result_name, int64_t axis,
                            const SunMMIOValue &valid_extent,
                            const SunMMIOType &tile_type);

  SunMMIOValue TileMaskAnd(const std::string &result_name,
                           const SunMMIOValue &lhs, const SunMMIOValue &rhs,
                           const SunMMIOType &tile_type);

  SunMMIOValue TileSelect(const std::string &result_name,
                          const SunMMIOValue &mask,
                          const SunMMIOValue &true_value,
                          const SunMMIOValue &false_value,
                          const SunMMIOType &result_type, DataType dtype);

  SunMMIOValue TileReduce(const std::string &result_name,
                          const std::string &predicate,
                          const SunMMIOValue &data,
                          const SunMMIOType &result_type, int64_t axis,
                          DataType dtype);

  SunMMIOValue TileSqueeze(const std::string &result_name,
                           const SunMMIOValue &tile,
                           const SunMMIOType &tile_type, int64_t axis,
                           DataType dtype);

  void TileStore(const SunMMIOValue &value, const SunMMIOValue &tile_view,
                 const std::optional<SunMMIOValue> &mask);

private:
  SunmmioMlirContext &ctx_;
  SunmmioMlirType type_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_TILE_OP_H_
