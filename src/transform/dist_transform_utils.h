/*!
 * \file tl/transform/dist_transform_utils.h
 * \brief Shared utilities for Rank-level communication passes.
 */

#ifndef TVM_TL_TRANSFORM_DIST_TRANSFORM_UTILS_H_
#define TVM_TL_TRANSFORM_DIST_TRANSFORM_UTILS_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../op/dist_comm.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {
namespace dist_transform {

using namespace tir;

inline constexpr const char *kDistWorldSizeAttr = "tl.dist.world_size";
inline constexpr const char *kDistSignalCountsAttr = "tl.dist.signal_counts";
inline constexpr const char *kAutoSignalKind = "auto";
inline constexpr int kIncrementFlagregCount = 8;
inline constexpr int kValueFlagregCount = 32;

enum class DistSignalScope { kSram, kDram };
enum class DistSignalUpdateMode { kIncrement, kValue, kMemory };
enum class DistSignalKind {
  kSramFlagregInc,
  kDramFlagregInc,
  kSramFlagregValue,
  kDramFlagregValue,
  kSramMemory,
  kDramMemory,
  kCount,
};

struct DistSignalKindInfo {
  DistSignalKind kind;
  const char *name;
  DistSignalScope scope;
  DistSignalUpdateMode update_mode;
  int capacity;
  DataType state_dtype;
  bool allow_multi_sender;
};

inline const std::array<DistSignalKindInfo,
                        static_cast<size_t>(DistSignalKind::kCount)> &
DistSignalKindInfos() {
  static const std::array<DistSignalKindInfo,
                          static_cast<size_t>(DistSignalKind::kCount)>
      infos{{
          {DistSignalKind::kSramFlagregInc, "sram_flagreg_inc",
           DistSignalScope::kSram, DistSignalUpdateMode::kIncrement,
           kIncrementFlagregCount, DataType::UInt(8), true},
          {DistSignalKind::kDramFlagregInc, "dram_flagreg_inc",
           DistSignalScope::kDram, DistSignalUpdateMode::kIncrement,
           kIncrementFlagregCount, DataType::UInt(8), true},
          {DistSignalKind::kSramFlagregValue, "sram_flagreg_value",
           DistSignalScope::kSram, DistSignalUpdateMode::kValue,
           kValueFlagregCount, DataType::UInt(32), false},
          {DistSignalKind::kDramFlagregValue, "dram_flagreg_value",
           DistSignalScope::kDram, DistSignalUpdateMode::kValue,
           kValueFlagregCount, DataType::UInt(32), false},
          {DistSignalKind::kSramMemory, "sram_memory", DistSignalScope::kSram,
           DistSignalUpdateMode::kMemory, -1, DataType::UInt(32), false},
          {DistSignalKind::kDramMemory, "dram_memory", DistSignalScope::kDram,
           DistSignalUpdateMode::kMemory, -1, DataType::UInt(32), false},
      }};
  return infos;
}

inline const DistSignalKindInfo *
FindDistSignalKindInfo(const std::string &name) {
  for (const DistSignalKindInfo &info : DistSignalKindInfos()) {
    if (name == info.name) {
      return &info;
    }
  }
  return nullptr;
}

inline std::string RequireStringImm(const PrimExpr &expr, const char *name) {
  const auto *imm = expr.as<StringImmNode>();
  ICHECK(imm) << name << " must be a compile-time string, got " << expr;
  return imm->value;
}

inline const DistSignalKindInfo &RequireDistSignalKindInfo(const PrimExpr &expr,
                                                           const char *name) {
  std::string kind_name = RequireStringImm(expr, name);
  const DistSignalKindInfo *info = FindDistSignalKindInfo(kind_name);
  ICHECK(info) << "Unsupported T.dist signal kind " << kind_name;
  return *info;
}

inline size_t DistSignalKindIndex(DistSignalKind kind) {
  return static_cast<size_t>(kind);
}

inline bool IsHighDistOp(const CallNode *call) {
  return call->op.same_as(dist_signal_decl()) ||
         call->op.same_as(dist_signal()) ||
         call->op.same_as(DistPutOp::Get()) ||
         call->op.same_as(DistPeerPutOp::Get()) ||
         call->op.same_as(DistRoutedPeerPutOp::Get()) ||
         call->op.same_as(DistWaitSignalOp::Get()) ||
         call->op.same_as(dist_wait_all()) ||
         call->op.same_as(dist_wait_send()) ||
         call->op.same_as(dist_expect()) ||
         call->op.same_as(dist_rank_routed_put()) ||
         call->op.same_as(dist_routed_put());
}

inline bool IsLeafDistOp(const CallNode *call) {
  return call->op.same_as(dist_put_()) ||
         call->op.same_as(dist_wait_signal_()) ||
         call->op.same_as(dist_wait_send()) || call->op.same_as(dist_expect_());
}

class DistOpDetector : public StmtExprVisitor {
public:
  explicit DistOpDetector(bool high_level) : high_level_(high_level) {}

  bool Detect(const Stmt &stmt) {
    VisitStmt(stmt);
    return found_;
  }

private:
  void VisitExpr_(const CallNode *call) final {
    if ((high_level_ && IsHighDistOp(call)) ||
        (!high_level_ && IsLeafDistOp(call))) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  bool high_level_{false};
  bool found_{false};
};

inline int64_t RequireIntImm(const PrimExpr &expr, const char *name) {
  const auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << name << " must be a compile-time integer, got " << expr;
  return imm->value;
}

inline bool IsDramScope(const ffi::String &scope) {
  return scope.empty() || scope == "global";
}

inline bool IsMultiRank(const PrimFunc &func) {
  auto world_size = func->GetAttr<Integer>(kDistWorldSizeAttr);
  return world_size && world_size.value()->value > 1;
}

class RouteExprBufferLoadDetector : public ExprVisitor {
public:
  bool Detect(const PrimExpr &expr) {
    VisitExpr(expr);
    return found_;
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    found_ = true;
    ExprVisitor::VisitExpr_(op);
  }

  bool found_{false};
};

struct NormalRouteEntry {
  int64_t origin_src_row;
  PrimExpr dst_rank;
  PrimExpr dst_row;
};

struct PeerRouteEntry {
  PrimExpr peer_row;
  PrimExpr dst_rank;
};

inline std::vector<NormalRouteEntry>
ParseNormalRouteTable(const PrimExpr &expr) {
  const auto *table = expr.as<CallNode>();
  ICHECK(table && table->op.same_as(dist_route_table()))
      << "Expected a T.dist route table, got " << expr;
  std::vector<NormalRouteEntry> routes;
  for (const PrimExpr &entry_expr : table->args) {
    const auto *entry = entry_expr.as<CallNode>();
    ICHECK(entry && entry->op.same_as(dist_route()));
    ICHECK_EQ(entry->args.size(), 3U);
    routes.push_back(NormalRouteEntry{
        RequireIntImm(entry->args[0], "route origin source row"),
        entry->args[1], entry->args[2]});
  }
  return routes;
}

inline std::vector<PeerRouteEntry> ParsePeerRouteTable(const PrimExpr &expr) {
  const auto *table = expr.as<CallNode>();
  ICHECK(table && table->op.same_as(dist_peer_route_table()))
      << "Expected a T.dist peer route table, got " << expr;
  std::vector<PeerRouteEntry> routes;
  for (const PrimExpr &entry_expr : table->args) {
    const auto *entry = entry_expr.as<CallNode>();
    ICHECK(entry && entry->op.same_as(dist_peer_route()));
    ICHECK_EQ(entry->args.size(), 2U);
    routes.push_back(PeerRouteEntry{entry->args[0], entry->args[1]});
  }
  return routes;
}

struct DistPassContext {
  Target target;
  int64_t world_size;
  Var rank_id;
};

inline Optional<DistPassContext> GetDistPassContext(const PrimFunc &func) {
  auto world_size = func->GetAttr<Integer>(kDistWorldSizeAttr);
  if (!world_size || world_size.value()->value <= 1) {
    return std::nullopt;
  }
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target && TargetIsSunmmio(target.value()));
  auto rank_index = func->GetAttr<Integer>("tl.dist.rank_id_param_index");
  ICHECK(rank_index) << "T.dist passes require a T.dist.RankId parameter";
  int64_t index = rank_index.value()->value;
  ICHECK_GE(index, 0);
  ICHECK_LT(index, static_cast<int64_t>(func->params.size()));
  return DistPassContext{target.value(), world_size.value()->value,
                         func->params[index]};
}

class DistRouteMutatorBase : public arith::IRMutatorWithAnalyzer {
public:
  DistRouteMutatorBase(arith::Analyzer *analyzer, Target target,
                       int64_t world_size, Var rank_id)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(std::move(target)),
        world_size_(world_size), rank_id_(std::move(rank_id)) {
    auto mesh = GetSunmmioMeshConfig(target_);
    mesh_nrows_ = mesh.nrow;
    mesh_ncols_ = mesh.ncol;
  }

protected:
  PrimExpr I32(int64_t value) const { return IntImm(DataType::Int(32), value); }

  PrimExpr CurrentRow(const PrimExpr &current_core) {
    return analyzer_->Simplify(floordiv(current_core, I32(mesh_ncols_)));
  }

  PrimExpr CurrentCol(const PrimExpr &current_core) {
    return analyzer_->Simplify(floormod(current_core, I32(mesh_ncols_)));
  }

  Map<Var, PrimExpr>
  MakeSourceSubstitution(const Var &current_core, int64_t src_row,
                         std::optional<int64_t> src_rank = std::nullopt) {
    PrimExpr source_core =
        I32(src_row * mesh_ncols_) + CurrentCol(current_core);
    Map<Var, PrimExpr> substitution{{current_core, source_core}};
    if (src_rank) {
      substitution.Set(rank_id_, I32(src_rank.value()));
    }
    return substitution;
  }

  PrimExpr RewriteForSource(const PrimExpr &expr, const Var &current_core,
                            int64_t src_row,
                            std::optional<int64_t> src_rank = std::nullopt) {
    return analyzer_->Simplify(Substitute(
        expr, MakeSourceSubstitution(current_core, src_row, src_rank)));
  }

  void ValidateRouteExpression(const PrimExpr &expr, const char *name) {
    ICHECK(!RouteExprBufferLoadDetector().Detect(expr))
        << "T.dist.put " << name
        << " cannot depend on BufferLoad in automatic row routing";
  }

  void ValidateResolvedEndpoint(const PrimExpr &dst_rank,
                                const PrimExpr &dst_row,
                                const char *op_name = "T.dist.put") {
    ICHECK(analyzer_->CanProve(dst_rank >= I32(0)) &&
           analyzer_->CanProve(dst_rank < I32(world_size_)))
        << op_name << " cannot prove dst_rank is in [0, " << world_size_
        << "): " << dst_rank;
    ICHECK(analyzer_->CanProve(dst_row >= I32(0)) &&
           analyzer_->CanProve(dst_row < I32(mesh_nrows_)))
        << op_name << " cannot prove dst_row is in [0, " << mesh_nrows_
        << "): " << dst_row;
  }

  bool IsRowInvariantPredicate(const PrimExpr &predicate,
                               const Var &current_core) {
    if (RouteExprBufferLoadDetector().Detect(predicate)) {
      return false;
    }
    PrimExpr baseline = RewriteForSource(predicate, current_core, 0);
    for (int64_t row = 1; row < mesh_nrows_; ++row) {
      PrimExpr candidate = RewriteForSource(predicate, current_core, row);
      if (!analyzer_->CanProve(baseline == candidate)) {
        return false;
      }
    }
    return true;
  }

  Target target_;
  int64_t world_size_;
  Var rank_id_;
  int mesh_nrows_{1};
  int mesh_ncols_{1};
};

} // namespace dist_transform
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_DIST_TRANSFORM_UTILS_H_
