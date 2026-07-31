/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file inject_sunmmio_sync.cc
 * \brief Inject synchronization primitives for SUNMMIO.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"
#include "./common/attr.h"
#include "./common/collector.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

bool IsSyncTokenExpr(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  if (!call) {
    return false;
  }
  return call->op.same_as(sync_token_id());
}

PrimExpr I64Imm(int64_t value) { return IntImm(DataType::Int(64), value); }

PrimExpr AsI64(PrimExpr value) {
  if (const auto *imm = value.as<IntImmNode>()) {
    return I64Imm(imm->value);
  }
  if (value.dtype() == DataType::Int(64)) {
    return value;
  }
  return Cast(DataType::Int(64), value);
}

PrimExpr CoreBitMask(PrimExpr core_id) {
  if (const auto *imm = core_id.as<IntImmNode>()) {
    ICHECK_GE(imm->value, 0);
    ICHECK_LT(imm->value, 64)
        << "barrier mask currently supports core ids in [0, 64)";
    return I64Imm(static_cast<int64_t>(uint64_t{1} << imm->value));
  }
  return I64Imm(1) << AsI64(core_id);
}

PrimExpr FullCoreMask(int total_cores) {
  ICHECK_GE(total_cores, 0);
  ICHECK_LE(total_cores, 64)
      << "barrier mask currently supports at most 64 cores";
  uint64_t mask =
      total_cores == 64 ? ~uint64_t{0} : ((uint64_t{1} << total_cores) - 1);
  return I64Imm(static_cast<int64_t>(mask));
}

bool SamePrimExpr(const PrimExpr &lhs, const PrimExpr &rhs) {
  return StructuralEqual()(lhs, rhs);
}

struct BarrierMaskInfo {
  PrimExpr expr;
  std::vector<int64_t> candidates;
};

using TokenBarrierMap = std::map<int, BarrierMaskInfo>;

void AddUniqueInt64(std::vector<int64_t> *values, int64_t value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

uint64_t UnsignedMask(int64_t value) { return static_cast<uint64_t>(value); }

std::optional<int64_t> FloorDivInt64(int64_t lhs, int64_t rhs) {
  if (rhs == 0) {
    return std::nullopt;
  }
  int64_t quotient = lhs / rhs;
  int64_t remainder = lhs % rhs;
  if (remainder != 0 && ((remainder > 0) != (rhs > 0))) {
    --quotient;
  }
  return quotient;
}

std::optional<int64_t> EvalInt64(PrimExpr expr, arith::Analyzer *analyzer) {
  if (analyzer) {
    expr = analyzer->Simplify(expr);
  }
  if (const auto *imm = expr.as<IntImmNode>()) {
    return static_cast<int64_t>(imm->value);
  }
  if (const auto *op = expr.as<CastNode>()) {
    return EvalInt64(op->value, analyzer);
  }

  auto eval_binary = [&](const PrimExpr &a, const PrimExpr &b,
                         auto fn) -> std::optional<int64_t> {
    std::optional<int64_t> lhs = EvalInt64(a, analyzer);
    std::optional<int64_t> rhs = EvalInt64(b, analyzer);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return fn(*lhs, *rhs);
  };

  if (const auto *op = expr.as<AddNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a + b; });
  }
  if (const auto *op = expr.as<SubNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a - b; });
  }
  if (const auto *op = expr.as<MulNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a * b; });
  }
  if (const auto *op = expr.as<DivNode>()) {
    return eval_binary(op->a, op->b, [](int64_t a, int64_t b) {
      return b == 0 ? std::optional<int64_t>() : std::optional<int64_t>(a / b);
    });
  }
  if (const auto *op = expr.as<ModNode>()) {
    return eval_binary(op->a, op->b, [](int64_t a, int64_t b) {
      return b == 0 ? std::optional<int64_t>() : std::optional<int64_t>(a % b);
    });
  }
  if (const auto *op = expr.as<FloorDivNode>()) {
    return eval_binary(
        op->a, op->b, [](int64_t a, int64_t b) { return FloorDivInt64(a, b); });
  }
  if (const auto *op = expr.as<FloorModNode>()) {
    return eval_binary(op->a, op->b, [](int64_t a, int64_t b) {
      std::optional<int64_t> div = FloorDivInt64(a, b);
      if (!div) {
        return std::optional<int64_t>();
      }
      return std::optional<int64_t>(a - (*div) * b);
    });
  }
  if (const auto *op = expr.as<EQNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a == b ? 1 : 0; });
  }
  if (const auto *op = expr.as<NENode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a != b ? 1 : 0; });
  }
  if (const auto *op = expr.as<LTNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a < b ? 1 : 0; });
  }
  if (const auto *op = expr.as<LENode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a <= b ? 1 : 0; });
  }
  if (const auto *op = expr.as<GTNode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a > b ? 1 : 0; });
  }
  if (const auto *op = expr.as<GENode>()) {
    return eval_binary(op->a, op->b,
                       [](int64_t a, int64_t b) { return a >= b ? 1 : 0; });
  }
  if (const auto *op = expr.as<SelectNode>()) {
    std::optional<int64_t> cond = EvalInt64(op->condition, analyzer);
    if (!cond) {
      return std::nullopt;
    }
    return EvalInt64(*cond != 0 ? op->true_value : op->false_value, analyzer);
  }
  if (const auto *call = expr.as<CallNode>()) {
    const auto *op = call->op.as<OpNode>();
    if (!op || call->args.size() != 2) {
      return std::nullopt;
    }
    const std::string name = op->name;
    if (name == "tir.bitwise_or") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) | UnsignedMask(b));
          });
    }
    if (name == "tir.bitwise_and") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) & UnsignedMask(b));
          });
    }
    if (name == "tir.bitwise_xor") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) ^ UnsignedMask(b));
          });
    }
    if (name == "tir.shift_left") {
      return eval_binary(call->args[0], call->args[1],
                         [](int64_t a, int64_t b) {
                           if (b < 0 || b >= 64) {
                             return std::optional<int64_t>();
                           }
                           return std::optional<int64_t>(
                               static_cast<int64_t>(UnsignedMask(a) << b));
                         });
    }
  }
  return std::nullopt;
}

int CountMaskBits(uint64_t mask) {
  return static_cast<int>(__builtin_popcountll(mask));
}

bool IsMaskWithinMesh(uint64_t mask, int total_cores) {
  if (total_cores == 64) {
    return true;
  }
  uint64_t full_mask = (uint64_t{1} << total_cores) - 1;
  return (mask & ~full_mask) == 0;
}

void AppendCandidates(std::vector<int64_t> *dst,
                      const std::vector<int64_t> &src) {
  for (int64_t mask : src) {
    AddUniqueInt64(dst, mask);
  }
}

bool MaskAlignedWithDirection(uint64_t mask, int direction, int mesh_nrow,
                              int mesh_ncol) {
  int total_cores = mesh_nrow * mesh_ncol;
  if (mask == 0 || !IsMaskWithinMesh(mask, total_cores)) {
    return false;
  }
  int bit_count = CountMaskBits(mask);
  int min_participants =
      direction == 0 ? std::min(mesh_ncol, 2) : std::min(mesh_nrow, 2);
  if (bit_count < min_participants) {
    return false;
  }

  int ref_row = -1;
  int ref_col = -1;
  for (int core = 0; core < total_cores; ++core) {
    if ((mask & (uint64_t{1} << core)) == 0) {
      continue;
    }
    int row = core / mesh_ncol;
    int col = core % mesh_ncol;
    if (ref_row < 0) {
      ref_row = row;
      ref_col = col;
      continue;
    }
    if (direction == 0 && row != ref_row) {
      return false;
    }
    if (direction == 1 && col != ref_col) {
      return false;
    }
  }
  return true;
}

class VarCollector : public ExprVisitor {
public:
  void VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    for (const Var &existing : vars) {
      if (existing.same_as(var)) {
        return;
      }
    }
    vars.push_back(var);
  }

  std::vector<Var> vars;
};

bool ExprUsesAnyVar(const PrimExpr &expr, const std::vector<Var> &vars) {
  if (vars.empty()) {
    return false;
  }
  VarCollector collector;
  collector(expr);
  for (const Var &used : collector.vars) {
    for (const Var &candidate : vars) {
      if (used.same_as(candidate)) {
        return true;
      }
    }
  }
  return false;
}

class UnsafeLoopDomainExprDetector : public ExprVisitor {
public:
  bool HasUnsafeExpr() const { return has_unsafe_expr_; }

  void VisitExpr_(const VarNode *op) final {
    if (op->dtype.is_handle()) {
      has_unsafe_expr_ = true;
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final { has_unsafe_expr_ = true; }

  void VisitExpr_(const CallNode *op) final {
    if (op->dtype.is_handle()) {
      has_unsafe_expr_ = true;
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }

private:
  bool has_unsafe_expr_{false};
};

bool CanUseLoopDomainForRegionCover(const PrimExpr &expr) {
  if (!expr.defined() || expr.dtype().is_handle()) {
    return false;
  }
  UnsafeLoopDomainExprDetector detector;
  detector(expr);
  return !detector.HasUnsafeExpr();
}

std::vector<int64_t> EnumerateMaskCandidates(PrimExpr expr, int direction,
                                             int mesh_nrow, int mesh_ncol,
                                             arith::Analyzer *analyzer) {
  VarCollector collector;
  collector(expr);
  if (collector.vars.empty()) {
    std::optional<int64_t> value = EvalInt64(expr, analyzer);
    if (value) {
      uint64_t mask = UnsignedMask(*value);
      if (MaskAlignedWithDirection(mask, direction, mesh_nrow, mesh_ncol)) {
        return {static_cast<int64_t>(mask)};
      }
    }
    return {};
  }
  if (collector.vars.size() > 2) {
    return {};
  }

  int total_cores = mesh_nrow * mesh_ncol;
  std::vector<int64_t> candidates;
  int64_t num_cases = 1;
  for (size_t i = 0; i < collector.vars.size(); ++i) {
    num_cases *= total_cores;
  }
  for (int64_t case_id = 0; case_id < num_cases; ++case_id) {
    Map<Var, PrimExpr> var_map;
    int64_t case_value = case_id;
    for (const Var &var : collector.vars) {
      int core = static_cast<int>(case_value % total_cores);
      case_value /= total_cores;
      var_map.Set(var, IntImm(var.dtype(), core));
    }
    PrimExpr candidate_expr = Substitute(expr, var_map);
    if (analyzer) {
      candidate_expr = analyzer->Simplify(candidate_expr);
    }
    std::optional<int64_t> value = EvalInt64(candidate_expr, analyzer);
    if (!value) {
      return {};
    }
    uint64_t mask = UnsignedMask(*value);
    if (MaskAlignedWithDirection(mask, direction, mesh_nrow, mesh_ncol)) {
      AddUniqueInt64(&candidates, static_cast<int64_t>(mask));
    }
  }
  return candidates;
}

Array<PrimExpr> MakeBarrierArgs(const BarrierMaskInfo &info) {
  Array<PrimExpr> args;
  args.push_back(info.expr);
  for (int64_t mask : info.candidates) {
    args.push_back(I64Imm(mask));
  }
  return args;
}

Array<PrimExpr> MakeBarrierInitArgs(const BarrierMaskInfo &info) {
  if (info.candidates.empty()) {
    return MakeBarrierArgs(info);
  }

  Array<PrimExpr> args;
  args.push_back(I64Imm(-1));
  for (int64_t mask : info.candidates) {
    args.push_back(I64Imm(mask));
  }
  return args;
}

BarrierMaskInfo BarrierMaskInfoFromArgs(const Array<PrimExpr> &args) {
  ICHECK_GE(args.size(), 1U) << "barrier call requires participant_mask";
  BarrierMaskInfo info;
  info.expr = args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    const auto *imm = args[i].as<IntImmNode>();
    ICHECK(imm) << "barrier candidate masks must be IntImm";
    AddUniqueInt64(&info.candidates, static_cast<int64_t>(imm->value));
  }
  return info;
}

bool SameBarrierMaskInfo(const BarrierMaskInfo &lhs,
                         const BarrierMaskInfo &rhs) {
  if (!SamePrimExpr(lhs.expr, rhs.expr)) {
    return false;
  }
  if (lhs.candidates.size() != rhs.candidates.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.candidates.size(); ++i) {
    if (lhs.candidates[i] != rhs.candidates[i]) {
      return false;
    }
  }
  return true;
}

bool BarrierCallMatchesInfo(const CallNode *call, const BarrierMaskInfo &info) {
  if (!call || !call->op.same_as(barrier_arrive_and_wait()) ||
      call->args.empty()) {
    return false;
  }
  return SameBarrierMaskInfo(BarrierMaskInfoFromArgs(call->args), info);
}

void AddUniqueBarrierMaskInfo(std::vector<BarrierMaskInfo> *values,
                              const BarrierMaskInfo &value) {
  for (const BarrierMaskInfo &existing : *values) {
    if (SameBarrierMaskInfo(existing, value)) {
      return;
    }
  }
  values->push_back(value);
}

bool BroadcastCallHasSrcCore(const CallNode *call) {
  ICHECK_GE(call->args.size(), static_cast<size_t>(kBroadcastArgCount))
      << "broadcast_() call is missing its fixed argument prefix.";
  size_t non_token_args = call->args.size();
  if (non_token_args > 0 && IsSyncTokenExpr(call->args.back())) {
    --non_token_args;
  }
  ICHECK(non_token_args == static_cast<size_t>(kBroadcastArgCount) ||
         non_token_args == static_cast<size_t>(kBroadcastArgCount + 1))
      << "broadcast_() expects fixed args plus optional src_core, got "
      << non_token_args << " non-token args.";
  return non_token_args == static_cast<size_t>(kBroadcastArgCount + 1);
}

PrimExpr GetBroadcastSrcCore(const CallNode *call) {
  ICHECK(BroadcastCallHasSrcCore(call))
      << "broadcast_() call does not carry optional src_core.";
  size_t non_token_args = call->args.size();
  if (IsSyncTokenExpr(call->args.back())) {
    --non_token_args;
  }
  return call->args[non_token_args - 1];
}

// Helper function to check if two memory regions intersect.
// Used for dependency analysis to determine if synchronization is needed.
bool RegionIntersect(const Region &region1, const Region &region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

Region
CoverRegionWithLoopDomains(const Region &region,
                           const ffi::Map<Var, arith::IntSet> &loop_domains) {
  if (loop_domains.empty()) {
    return region;
  }

  ffi::Array<arith::IntSet> relaxed = arith::EvalSet(region, loop_domains);
  ICHECK_EQ(relaxed.size(), region.size());
  Region covered;
  covered.reserve(relaxed.size());
  arith::Analyzer analyzer;
  for (size_t i = 0; i < relaxed.size(); ++i) {
    PrimExpr min = analyzer.Simplify(relaxed[i].min());
    PrimExpr extent = analyzer.Simplify(relaxed[i].max() - min + 1);
    covered.push_back(Range::FromMinExtent(min, extent));
  }
  return covered;
}

// Visitor to collect all buffer read and write accesses within an expression or
// statement. This is used to identify what memory is being touched.
class BufferAccessCollector : public ExprVisitor {
public:
  BufferAccessCollector(Map<Var, Buffer> buffer_data_to_buffer,
                        ffi::Map<Var, arith::IntSet> loop_domains = {})
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        loop_domains_(std::move(loop_domains)) {}

  Array<BufferRegion> GetReads() const { return reads_; }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    auto load_buffer = op->buffer;
    Array<PrimExpr> indices = op->indices;
    // convert indices to region
    Array<Range> region;
    for (const auto &index : indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    auto load_region = BufferRegion(
        load_buffer, CoverRegionWithLoopDomains(region, loop_domains_));
    reads_.push_back(load_region);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    auto args = op->args;
    if (op->op.same_as(builtin::address_of())) {
      BufferRegion buffer_region;
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        buffer_region = BufferRegion::FullRegion(load->buffer);
        for (const auto &index : load->indices) {
          VisitExpr(index);
        }
      } else if (const auto *var_node = op->args[0].as<VarNode>()) {
        Var data_var = tvm::ffi::GetRef<Var>(var_node);
        auto it = buffer_data_to_buffer_.find(data_var);
        if (it != buffer_data_to_buffer_.end()) {
          buffer_region = BufferRegion::FullRegion((*it).second);
        }
      }
      if (buffer_region.defined()) {
        reads_.push_back(buffer_region);
      }
    } else if (op->op.same_as(builtin::tvm_access_ptr())) {
      const VarNode *buffer_var = op->args[1].as<VarNode>();
      ICHECK(buffer_var);
      auto it = buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(buffer_var));
      if (it != buffer_data_to_buffer_.end()) {
        const Buffer &buffer = (*it).second;
        const BufferRegion buffer_region = BufferRegion::FullRegion(buffer);
        reads_.push_back(buffer_region);
      }
      ExprVisitor::VisitExpr_(op);
    } else {
      ExprVisitor::VisitExpr_(op);
    }
  }

private:
  Array<BufferRegion> reads_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  ffi::Map<Var, arith::IntSet> loop_domains_;
};

// Collector for asynchronous operations within a loop body.
// Identifies DMA copies, layout transforms, MMA operations, and Broadcasts that
// happen asynchronously.
struct AccessRecord {
  Buffer buffer;
  Region region;
};

struct AsyncOpRecord {
  const EvaluateNode *op{nullptr};
  const CallNode *call{nullptr};
  int token{-1};
  int order{-1};
  std::vector<AccessRecord> reads;
  std::vector<AccessRecord> writes;
};

struct SyncAccessRecord {
  const StmtNode *stmt{nullptr};
  int order{-1};
  std::vector<AccessRecord> reads;
  std::vector<AccessRecord> writes;
};

class LoopAsyncCollector : public StmtVisitor {
public:
  explicit LoopAsyncCollector(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)) {}

  void VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = op->value.as<CallNode>();
    if (call) {
      AsyncOpRecord rec;
      rec.op = op;
      rec.call = call;
      if (call->op.same_as(dma_copy())) {
        rec.order = order_++;
        auto src = NormalizeToBufferRegion(call->args[0]);
        auto dst = NormalizeToBufferRegion(call->args[1]);
        rec.reads.push_back(AccessFromRegion(src));
        AppendAsyncArgumentReads(call, {0, 1}, &rec.reads);
        rec.writes.push_back(AccessFromRegion(dst));
        async_ops.push_back(rec);
      } else if (call->op.same_as(sunmmio_layout_transform())) {
        rec.order = order_++;
        auto src = NormalizeToBufferRegion(call->args[0]);
        auto dst = NormalizeToBufferRegion(call->args[1]);
        rec.reads.push_back(AccessFromRegion(src));
        AppendAsyncArgumentReads(call, {0, 1}, &rec.reads);
        rec.writes.push_back(AccessFromRegion(dst));
        async_ops.push_back(rec);
      } else if (call->op.same_as(mma_sunmmio())) {
        rec.order = order_++;
        auto lhs = NormalizeToBufferRegion(call->args[0]);
        auto rhs = NormalizeToBufferRegion(call->args[1]);
        auto acc = NormalizeToBufferRegion(call->args[2]);
        rec.reads.push_back(AccessFromRegion(lhs));
        rec.reads.push_back(AccessFromRegion(rhs));
        rec.reads.push_back(AccessFromRegion(acc));
        AppendAsyncArgumentReads(call, {0, 1, 2}, &rec.reads);
        rec.writes.push_back(AccessFromRegion(acc));
        async_ops.push_back(rec);
      } else if (call->op.same_as(broadcast_())) {
        rec.order = order_++;
        auto src = NormalizeToBufferRegion(call->args[0]);
        auto dst = NormalizeToBufferRegion(call->args[1]);
        rec.reads.push_back(AccessFromRegion(src));
        AppendAsyncArgumentReads(call, {0, 1}, &rec.reads);
        rec.writes.push_back(AccessFromRegion(dst));
        async_ops.push_back(rec);
      }
      if (!rec.reads.empty() || !rec.writes.empty()) {
        StmtVisitor::VisitStmt_(op);
        return;
      }
    }
    RecordSyncAccess(op, {op->value});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    RecordSyncAccess(op, {op->value});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    RecordSyncAccess(op, {op->value});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileNode *op) final {
    RecordSyncAccess(op, {op->condition});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) final {
    RecordSyncAccess(op, {op->condition});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode *op) final {
    RecordSyncAccess(op, {op->condition});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AssertStmtNode *op) final {
    RecordSyncAccess(op, {op->condition, op->message});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockRealizeNode *op) final {
    RecordSyncAccess(op, {op->predicate});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Array<Range> region;
    region.reserve(op->indices.size());
    std::vector<PrimExpr> read_exprs;
    read_exprs.reserve(op->indices.size() + 1);
    for (const auto &index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
      read_exprs.push_back(index);
    }
    read_exprs.push_back(op->value);
    RecordSyncAccess(
        op, read_exprs,
        {{op->buffer, CoverRegionWithLoopDomains(region, loop_domains_)}});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    RecordSyncAccess(op, {op->condition});
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    RecordSyncAccess(op, {op->min, op->extent});
    ffi::Map<Var, arith::IntSet> old_loop_domains = loop_domains_;
    if (CanUseLoopDomainForRegionCover(op->min) &&
        CanUseLoopDomainForRegionCover(op->extent)) {
      loop_domains_.Set(
          op->loop_var,
          arith::IntSet::FromRange(Range::FromMinExtent(op->min, op->extent)));
    }
    StmtVisitor::VisitStmt_(op);
    loop_domains_ = std::move(old_loop_domains);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    StmtVisitor::VisitStmt_(op);
    for (const Buffer &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
  }

  std::vector<AsyncOpRecord> async_ops;
  std::vector<SyncAccessRecord> sync_accesses;
  std::unordered_map<const StmtNode *, int> stmt_order;

private:
  std::vector<AccessRecord> CollectReads(PrimExpr expr) const {
    std::vector<AccessRecord> reads;
    BufferAccessCollector collector(buffer_data_to_buffer_, loop_domains_);
    collector(expr);
    for (const BufferRegion &read : collector.GetReads()) {
      reads.push_back({read->buffer, read->region});
      for (const Range &range : read->region) {
        std::vector<AccessRecord> min_reads = CollectReads(range->min);
        reads.insert(reads.end(), min_reads.begin(), min_reads.end());
        std::vector<AccessRecord> extent_reads = CollectReads(range->extent);
        reads.insert(reads.end(), extent_reads.begin(), extent_reads.end());
      }
    }
    return reads;
  }

  AccessRecord AccessFromRegion(const BufferRegion &region) const {
    return {region->buffer,
            CoverRegionWithLoopDomains(region->region, loop_domains_)};
  }

  void AppendReadsFromExpr(const PrimExpr &expr,
                           std::vector<AccessRecord> *out) const {
    std::vector<AccessRecord> reads = CollectReads(expr);
    out->insert(out->end(), reads.begin(), reads.end());
  }

  void AppendRegionBoundReads(const BufferRegion &region,
                              std::vector<AccessRecord> *out) const {
    for (const Range &range : region->region) {
      AppendReadsFromExpr(range->min, out);
      AppendReadsFromExpr(range->extent, out);
    }
  }

  void AppendAsyncArgumentReads(const CallNode *call,
                                std::initializer_list<size_t> region_args,
                                std::vector<AccessRecord> *out) const {
    size_t arg_count = call->args.size();
    if (arg_count != 0 && IsSyncTokenExpr(call->args.back())) {
      --arg_count;
    }
    for (size_t i = 0; i < arg_count; ++i) {
      if (std::find(region_args.begin(), region_args.end(), i) !=
          region_args.end()) {
        AppendRegionBoundReads(NormalizeToBufferRegion(call->args[i]), out);
      } else {
        AppendReadsFromExpr(call->args[i], out);
      }
    }
  }

  void RecordSyncAccess(const StmtNode *stmt,
                        const std::vector<PrimExpr> &exprs,
                        std::vector<AccessRecord> writes) {
    SyncAccessRecord rec;
    rec.stmt = stmt;
    rec.order = order_++;
    rec.writes = std::move(writes);
    for (const PrimExpr &expr : exprs) {
      std::vector<AccessRecord> reads = CollectReads(expr);
      rec.reads.insert(rec.reads.end(), reads.begin(), reads.end());
    }
    if (rec.reads.empty() && rec.writes.empty()) {
      --order_;
      return;
    }
    stmt_order[stmt] = rec.order;
    sync_accesses.push_back(std::move(rec));
  }

  void RecordSyncAccess(
      const StmtNode *stmt, std::initializer_list<PrimExpr> exprs,
      std::vector<AccessRecord> writes = std::vector<AccessRecord>()) {
    RecordSyncAccess(stmt, std::vector<PrimExpr>(exprs), std::move(writes));
  }

  int order_{0};
  Map<Var, Buffer> buffer_data_to_buffer_;
  ffi::Map<Var, arith::IntSet> loop_domains_;
};

// Represents the scope of a loop for dependency tracking.
// Stores writes that happen within the loop to check for loop-carried
// dependencies.
struct LoopScope {
  Var loop_var;
  PrimExpr loop_min;
  PrimExpr loop_extent;
  std::vector<AsyncOpRecord> async_ops;
  std::vector<SyncAccessRecord> sync_accesses;
  std::map<int, std::set<int>> prev_iter_waits_by_curr_token;
  std::map<int, std::set<int>> prev_iter_waits_by_sync_order;
  std::set<int> loop_entry_null_tokens;
  std::map<int, const CallNode *> token_to_call;
  std::unordered_map<const StmtNode *, int> stmt_order;
};

// Main rewriter class to inject synchronization primitives.
// It tracks buffer accesses and inserts wait_token and barrier_wait calls
// to enforce correct ordering based on data dependencies.
class InjectSyncRewriter : public StmtMutator {
public:
  InjectSyncRewriter(Map<Var, Buffer> buffer_data_to_buffer,
                     const Target &target, arith::Analyzer *analyzer,
                     bool insert_barrier_wait_after_wait_token = false)
      : token_count(0), mesh_nrow_(0), mesh_ncol_(0), analyzer_(analyzer),
        insert_barrier_wait_after_wait_token_(
            insert_barrier_wait_after_wait_token),
        buffer_data_to_buffer_(buffer_data_to_buffer) {
    SunmmioMeshConfig mesh = GetSunmmioMeshConfig(target);
    mesh_nrow_ = mesh.nrow;
    mesh_ncol_ = mesh.ncol;
  }

  TokenBarrierMap get_token_to_barrier_mask() const {
    return token_to_barrier_mask_;
  }

private:
  Region ShiftRegionByIterDelta(const Region &region, const Var &loop_var,
                                int delta) const {
    if (!loop_var.defined()) {
      return region;
    }
    Map<Var, PrimExpr> var_map;
    var_map.Set(loop_var, loop_var + delta);
    Region shifted_region;
    shifted_region.reserve(region.size());
    for (const auto &range : region) {
      shifted_region.push_back(Range::FromMinExtent(
          Substitute(range->min, var_map), Substitute(range->extent, var_map)));
    }
    return shifted_region;
  }

  bool MayOverlapAcrossIterations(const Region &curr_region,
                                  const Region &prev_region,
                                  const LoopScope &scope) const {
    if (!scope.loop_var.defined()) {
      return false;
    }
    if (analyzer_ && analyzer_->CanProve(scope.loop_extent <= 1)) {
      return false;
    }
    return RegionIntersect(
        curr_region, ShiftRegionByIterDelta(prev_region, scope.loop_var, -1));
  }

  bool AccessMayDependAcrossIterations(const AccessRecord &prev_access,
                                       const AccessRecord &curr_access,
                                       const LoopScope &scope) const {
    if (!prev_access.buffer.same_as(curr_access.buffer)) {
      return false;
    }
    return MayOverlapAcrossIterations(curr_access.region, prev_access.region,
                                      scope);
  }

  bool AccessMayDependWithinIteration(const AccessRecord &prev_access,
                                      const AccessRecord &curr_access) const {
    if (!prev_access.buffer.same_as(curr_access.buffer)) {
      return false;
    }
    return RegionIntersect(curr_access.region, prev_access.region);
  }

  bool HasLoopCarriedDependence(const AsyncOpRecord &prev_op,
                                const AsyncOpRecord &curr_op,
                                const LoopScope &scope) const {
    if (prev_op.order < curr_op.order) {
      return false;
    }

    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_read : curr_op.reads) {
        if (AccessMayDependAcrossIterations(prev_write, curr_read, scope)) {
          return true;
        }
      }
    }
    for (const auto &prev_read : prev_op.reads) {
      for (const auto &curr_write : curr_op.writes) {
        if (AccessMayDependAcrossIterations(prev_read, curr_write, scope)) {
          return true;
        }
      }
    }
    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_write : curr_op.writes) {
        if (AccessMayDependAcrossIterations(prev_write, curr_write, scope)) {
          return true;
        }
      }
    }
    return false;
  }

  bool HasLoopCarriedDependence(const AsyncOpRecord &prev_op,
                                const SyncAccessRecord &curr_access,
                                const LoopScope &scope) const {
    if (prev_op.order < curr_access.order) {
      return false;
    }

    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_read : curr_access.reads) {
        if (AccessMayDependAcrossIterations(prev_write, curr_read, scope)) {
          return true;
        }
      }
    }
    for (const auto &prev_read : prev_op.reads) {
      for (const auto &curr_write : curr_access.writes) {
        if (AccessMayDependAcrossIterations(prev_read, curr_write, scope)) {
          return true;
        }
      }
    }
    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_write : curr_access.writes) {
        if (AccessMayDependAcrossIterations(prev_write, curr_write, scope)) {
          return true;
        }
      }
    }
    return false;
  }

  bool HasWhileLoopCarriedDependence(const AsyncOpRecord &prev_op,
                                     const AsyncOpRecord &curr_op) const {
    if (prev_op.order < curr_op.order) {
      return false;
    }

    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_read : curr_op.reads) {
        if (AccessMayDependWithinIteration(prev_write, curr_read)) {
          return true;
        }
      }
    }
    for (const auto &prev_read : prev_op.reads) {
      for (const auto &curr_write : curr_op.writes) {
        if (AccessMayDependWithinIteration(prev_read, curr_write)) {
          return true;
        }
      }
    }
    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_write : curr_op.writes) {
        if (AccessMayDependWithinIteration(prev_write, curr_write)) {
          return true;
        }
      }
    }
    return false;
  }

  bool
  HasWhileLoopCarriedDependence(const AsyncOpRecord &prev_op,
                                const SyncAccessRecord &curr_access) const {
    if (prev_op.order < curr_access.order) {
      return false;
    }

    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_read : curr_access.reads) {
        if (AccessMayDependWithinIteration(prev_write, curr_read)) {
          return true;
        }
      }
    }
    for (const auto &prev_read : prev_op.reads) {
      for (const auto &curr_write : curr_access.writes) {
        if (AccessMayDependWithinIteration(prev_read, curr_write)) {
          return true;
        }
      }
    }
    for (const auto &prev_write : prev_op.writes) {
      for (const auto &curr_write : curr_access.writes) {
        if (AccessMayDependWithinIteration(prev_write, curr_write)) {
          return true;
        }
      }
    }
    return false;
  }

  bool HasIntraIterationDependentSuccessor(const AsyncOpRecord &producer,
                                           const LoopScope &scope) const {
    for (const auto &later_op : scope.async_ops) {
      if (later_op.order <= producer.order) {
        continue;
      }
      for (const auto &producer_write : producer.writes) {
        for (const auto &later_read : later_op.reads) {
          if (AccessMayDependWithinIteration(producer_write, later_read)) {
            return true;
          }
        }
      }
      for (const auto &producer_read : producer.reads) {
        for (const auto &later_write : later_op.writes) {
          if (AccessMayDependWithinIteration(producer_read, later_write)) {
            return true;
          }
        }
      }
      for (const auto &producer_write : producer.writes) {
        for (const auto &later_write : later_op.writes) {
          if (AccessMayDependWithinIteration(producer_write, later_write)) {
            return true;
          }
        }
      }
    }
    for (const auto &later_access : scope.sync_accesses) {
      if (later_access.order <= producer.order) {
        continue;
      }
      for (const auto &producer_write : producer.writes) {
        for (const auto &later_read : later_access.reads) {
          if (AccessMayDependWithinIteration(producer_write, later_read)) {
            return true;
          }
        }
      }
      for (const auto &producer_read : producer.reads) {
        for (const auto &later_write : later_access.writes) {
          if (AccessMayDependWithinIteration(producer_read, later_write)) {
            return true;
          }
        }
      }
      for (const auto &producer_write : producer.writes) {
        for (const auto &later_write : later_access.writes) {
          if (AccessMayDependWithinIteration(producer_write, later_write)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void AnalyzeLoopCarriedDependencies(LoopScope *scope) {
    if (!scope->loop_var.defined()) {
      return;
    }
    if (analyzer_ && analyzer_->CanProve(scope->loop_extent <= 1)) {
      return;
    }
    if (scope->async_ops.empty()) {
      return;
    }

    for (const auto &prev_op : scope->async_ops) {
      if (HasIntraIterationDependentSuccessor(prev_op, *scope)) {
        continue;
      }
      int consumer_token = -1;
      for (const auto &curr_op : scope->async_ops) {
        if (!HasLoopCarriedDependence(prev_op, curr_op, *scope)) {
          continue;
        }
        consumer_token = curr_op.token;
        break;
      }
      if (consumer_token >= 0) {
        scope->prev_iter_waits_by_curr_token[consumer_token].insert(
            prev_op.token);
        scope->loop_entry_null_tokens.insert(prev_op.token);
      }
    }

    for (const auto &prev_op : scope->async_ops) {
      if (HasIntraIterationDependentSuccessor(prev_op, *scope)) {
        continue;
      }
      for (const auto &curr_access : scope->sync_accesses) {
        if (!HasLoopCarriedDependence(prev_op, curr_access, *scope)) {
          continue;
        }
        scope->prev_iter_waits_by_sync_order[curr_access.order].insert(
            prev_op.token);
        scope->loop_entry_null_tokens.insert(prev_op.token);
      }
    }
  }

  void AnalyzeWhileLoopCarriedDependencies(LoopScope *scope) {
    if (scope->async_ops.empty()) {
      return;
    }

    for (const auto &prev_op : scope->async_ops) {
      if (HasIntraIterationDependentSuccessor(prev_op, *scope)) {
        continue;
      }
      int consumer_token = -1;
      for (const auto &curr_op : scope->async_ops) {
        if (!HasWhileLoopCarriedDependence(prev_op, curr_op)) {
          continue;
        }
        consumer_token = curr_op.token;
        break;
      }
      if (consumer_token >= 0) {
        scope->prev_iter_waits_by_curr_token[consumer_token].insert(
            prev_op.token);
        scope->loop_entry_null_tokens.insert(prev_op.token);
      }
    }

    for (const auto &prev_op : scope->async_ops) {
      if (HasIntraIterationDependentSuccessor(prev_op, *scope)) {
        continue;
      }
      for (const auto &curr_access : scope->sync_accesses) {
        if (!HasWhileLoopCarriedDependence(prev_op, curr_access)) {
          continue;
        }
        scope->prev_iter_waits_by_sync_order[curr_access.order].insert(
            prev_op.token);
        scope->loop_entry_null_tokens.insert(prev_op.token);
      }
    }
  }

  void InjectLoopEntryNullTokens(const LoopScope &scope, Array<Stmt> &stmts) {
    for (int token : scope.loop_entry_null_tokens) {
      stmts.push_back(Evaluate(Call(DataType::Handle(), sync_null_token(),
                                    {IntImm(DataType::Int(32), token)})));
      available_tokens_.insert(token);
    }
  }

  int PreAssignLoopToken(const AsyncOpRecord &async_op,
                         std::vector<const EvaluateNode *> *owned_ops) {
    auto it = pre_assigned_tokens_.find(async_op.op);
    if (it != pre_assigned_tokens_.end()) {
      return it->second;
    }

    int token = GetNextTokenId();
    pre_assigned_tokens_[async_op.op] = token;
    owned_ops->push_back(async_op.op);
    return token;
  }

  void ClearOwnedPreAssignedTokens(
      const std::vector<const EvaluateNode *> &owned_ops) {
    for (const EvaluateNode *op : owned_ops) {
      pre_assigned_tokens_.erase(op);
    }
  }

  PrimExpr LocalMaskBitSet(PrimExpr local_mask, int local_index) {
    PrimExpr bit = I64Imm(static_cast<int64_t>(uint64_t{1} << local_index));
    return (AsI64(local_mask) & bit) != I64Imm(0);
  }

  std::optional<int64_t>
  TryExpandBroadcastLocalMaskImm(const PrimExpr &local_mask, int direction,
                                 const PrimExpr &src_core) {
    std::optional<int64_t> local_value = EvalInt64(local_mask, analyzer_);
    std::optional<int64_t> src_value = EvalInt64(src_core, analyzer_);
    if (!local_value || !src_value) {
      return std::nullopt;
    }

    int total_cores = mesh_nrow_ * mesh_ncol_;
    ICHECK_GE(*src_value, 0);
    ICHECK_LT(*src_value, total_cores);

    int src_row = static_cast<int>(*src_value) / mesh_ncol_;
    int src_col = static_cast<int>(*src_value) % mesh_ncol_;
    int axis_len = direction == 0 ? mesh_ncol_ : mesh_nrow_;
    uint64_t valid_local_mask =
        axis_len == 64 ? ~uint64_t{0} : ((uint64_t{1} << axis_len) - 1);
    uint64_t local = UnsignedMask(*local_value);
    ICHECK_EQ(local & ~valid_local_mask, 0U)
        << "tl.broadcast_ direction-local mask has bits outside the active "
           "mesh axis";

    uint64_t global = 0;
    if (direction == 0) {
      for (int col = 0; col < mesh_ncol_; ++col) {
        if ((local & (uint64_t{1} << col)) != 0) {
          global |= uint64_t{1} << (src_row * mesh_ncol_ + col);
        }
      }
    } else {
      for (int row = 0; row < mesh_nrow_; ++row) {
        if ((local & (uint64_t{1} << row)) != 0) {
          global |= uint64_t{1} << (row * mesh_ncol_ + src_col);
        }
      }
    }
    return static_cast<int64_t>(global);
  }

  PrimExpr ExpandBroadcastLocalMask(const PrimExpr &local_mask, int direction,
                                    const PrimExpr &src_core) {
    if (std::optional<int64_t> imm =
            TryExpandBroadcastLocalMaskImm(local_mask, direction, src_core)) {
      return I64Imm(*imm);
    }

    PrimExpr src_core_i64 = AsI64(src_core);
    PrimExpr ncol = I64Imm(mesh_ncol_);
    PrimExpr src_row = floordiv(src_core_i64, ncol);
    PrimExpr src_col = floormod(src_core_i64, ncol);
    PrimExpr global_mask = I64Imm(0);

    if (direction == 0) {
      for (int col = 0; col < mesh_ncol_; ++col) {
        PrimExpr global_core = src_row * ncol + I64Imm(col);
        PrimExpr bit = CoreBitMask(global_core);
        global_mask = Select(LocalMaskBitSet(local_mask, col),
                             AsI64(global_mask) | AsI64(bit), global_mask);
      }
    } else {
      ICHECK_EQ(direction, 1)
          << "tl.broadcast_ local mask expansion only supports direction 0/1";
      for (int row = 0; row < mesh_nrow_; ++row) {
        PrimExpr global_core = I64Imm(row * mesh_ncol_) + src_col;
        PrimExpr bit = CoreBitMask(global_core);
        global_mask = Select(LocalMaskBitSet(local_mask, row),
                             AsI64(global_mask) | AsI64(bit), global_mask);
      }
    }

    return analyzer_ ? analyzer_->Simplify(global_mask) : global_mask;
  }

  PrimExpr GetBroadcastBarrierSrcCore(const CallNode *call) {
    if (BroadcastCallHasSrcCore(call)) {
      return GetBroadcastSrcCore(call);
    }
    ICHECK(current_kernel_core_id_.defined())
        << "tl.broadcast_ without optional src_core is treated as an "
           "all_gather leaf and requires an enclosing blockIdx.x binding";
    return current_kernel_core_id_;
  }

  // Inserts wait_token and, when enabled, the legacy paired barrier_wait.
  void process_wait_token_and_barrier_wait(Array<Stmt> &stmts, int token_id) {
    stmts.push_back(Evaluate(Call(DataType::Handle(), wait_token(),
                                  {IntImm(DataType::Int(32), token_id)})));
    auto barrier_it = token_to_barrier_mask_.find(token_id);
    if (insert_barrier_wait_after_wait_token_ &&
        barrier_it != token_to_barrier_mask_.end()) {
      stmts.push_back(
          Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                        MakeBarrierArgs(barrier_it->second))));
    }
  }

  bool IsWaitSuppressedByLoopHoist(int token_id) const {
    for (auto it = loop_hoisted_wait_tokens_stack_.rbegin();
         it != loop_hoisted_wait_tokens_stack_.rend(); ++it) {
      if (it->count(token_id) != 0) {
        return true;
      }
    }
    return false;
  }

  void maybe_process_wait_token_and_barrier_wait(Array<Stmt> &stmts,
                                                 int token_id) {
    if (IsWaitSuppressedByLoopHoist(token_id)) {
      return;
    }
    process_wait_token_and_barrier_wait(stmts, token_id);
  }

  void InjectLoopCarriedWaitsForToken(Array<Stmt> &stmts, int curr_token_id) {
    std::unordered_set<int> injected_tokens;
    for (int i = static_cast<int>(loop_scopes_.size()) - 1; i >= 0; --i) {
      auto it =
          loop_scopes_[i].prev_iter_waits_by_curr_token.find(curr_token_id);
      if (it == loop_scopes_[i].prev_iter_waits_by_curr_token.end()) {
        continue;
      }
      for (int token_id : it->second) {
        if (injected_tokens.count(token_id) != 0) {
          continue;
        }
        maybe_process_wait_token_and_barrier_wait(stmts, token_id);
        injected_tokens.insert(token_id);
      }
    }
  }

  void InjectLoopCarriedWaitsForSyncStmt(Array<Stmt> &stmts,
                                         const StmtNode *stmt) {
    std::unordered_set<int> injected_tokens;
    for (int i = static_cast<int>(loop_scopes_.size()) - 1; i >= 0; --i) {
      auto order_it = loop_scopes_[i].stmt_order.find(stmt);
      if (order_it == loop_scopes_[i].stmt_order.end()) {
        continue;
      }
      auto waits_it =
          loop_scopes_[i].prev_iter_waits_by_sync_order.find(order_it->second);
      if (waits_it == loop_scopes_[i].prev_iter_waits_by_sync_order.end()) {
        continue;
      }
      for (int token_id : waits_it->second) {
        if (injected_tokens.count(token_id) != 0) {
          continue;
        }
        maybe_process_wait_token_and_barrier_wait(stmts, token_id);
        injected_tokens.insert(token_id);
      }
    }
  }

  void CollectLoopCarriedWaitsForToken(int curr_token_id,
                                       std::set<int> *tokens) const {
    for (int i = static_cast<int>(loop_scopes_.size()) - 1; i >= 0; --i) {
      auto it =
          loop_scopes_[i].prev_iter_waits_by_curr_token.find(curr_token_id);
      if (it == loop_scopes_[i].prev_iter_waits_by_curr_token.end()) {
        continue;
      }
      tokens->insert(it->second.begin(), it->second.end());
    }
  }

  void CollectLoopCarriedWaitsForSyncStmt(const StmtNode *stmt,
                                          std::set<int> *tokens) const {
    for (int i = static_cast<int>(loop_scopes_.size()) - 1; i >= 0; --i) {
      auto order_it = loop_scopes_[i].stmt_order.find(stmt);
      if (order_it == loop_scopes_[i].stmt_order.end()) {
        continue;
      }
      auto waits_it =
          loop_scopes_[i].prev_iter_waits_by_sync_order.find(order_it->second);
      if (waits_it == loop_scopes_[i].prev_iter_waits_by_sync_order.end()) {
        continue;
      }
      tokens->insert(waits_it->second.begin(), waits_it->second.end());
    }
  }

  bool IsTokenHoistableBeforeLoop(
      int token_id, const std::set<int> &generated_tokens_in_loop) const {
    return token_id >= 0 && available_tokens_.count(token_id) != 0 &&
           generated_tokens_in_loop.count(token_id) == 0 &&
           !IsWaitSuppressedByLoopHoist(token_id);
  }

  AccessRecord CoverAccessForLoopHoist(const AccessRecord &access,
                                       const LoopScope &scope) const {
    if (!scope.loop_var.defined() || !scope.loop_min.defined() ||
        !scope.loop_extent.defined() ||
        !CanUseLoopDomainForRegionCover(scope.loop_min) ||
        !CanUseLoopDomainForRegionCover(scope.loop_extent)) {
      return access;
    }
    ffi::Map<Var, arith::IntSet> loop_domain;
    loop_domain.Set(scope.loop_var,
                    arith::IntSet::FromRange(Range::FromMinExtent(
                        scope.loop_min, scope.loop_extent)));
    return {access.buffer,
            CoverRegionWithLoopDomains(access.region, loop_domain)};
  }

  void AddHoistableToken(int token_id,
                         const std::set<int> &generated_tokens_in_loop,
                         std::set<int> *hoistable_tokens) const {
    if (IsTokenHoistableBeforeLoop(token_id, generated_tokens_in_loop)) {
      hoistable_tokens->insert(token_id);
    }
  }

  void CollectHoistableWaitsForReadAccess(
      const AccessRecord &read_access,
      const std::set<int> &generated_tokens_in_loop,
      std::set<int> *hoistable_tokens) const {
    for (const Array<ObjectRef> &buf : write_buffers) {
      Buffer buf_buffer = Downcast<Buffer>(buf[0]);
      Region buf_region = Downcast<Region>(buf[1]);
      if (read_access.buffer.same_as(buf_buffer) &&
          RegionIntersect(read_access.region, buf_region)) {
        AddHoistableToken(write_buffer_token_map[buf], generated_tokens_in_loop,
                          hoistable_tokens);
      }
    }
  }

  void CollectHoistableWaitsForWriteAccess(
      const AccessRecord &write_access,
      const std::set<int> &generated_tokens_in_loop,
      std::set<int> *hoistable_tokens) const {
    for (const Array<ObjectRef> &buf : read_buffers) {
      Buffer buf_buffer = Downcast<Buffer>(buf[0]);
      Region buf_region = Downcast<Region>(buf[1]);
      if (write_access.buffer.same_as(buf_buffer) &&
          RegionIntersect(write_access.region, buf_region)) {
        AddHoistableToken(read_buffer_token_map[buf], generated_tokens_in_loop,
                          hoistable_tokens);
      }
    }
    for (const Array<ObjectRef> &buf : write_buffers) {
      Buffer buf_buffer = Downcast<Buffer>(buf[0]);
      Region buf_region = Downcast<Region>(buf[1]);
      if (write_access.buffer.same_as(buf_buffer) &&
          RegionIntersect(write_access.region, buf_region)) {
        AddHoistableToken(write_buffer_token_map[buf], generated_tokens_in_loop,
                          hoistable_tokens);
      }
    }
  }

  std::set<int> AnalyzeHoistableWaitsForLoop(const LoopScope &scope) const {
    std::set<int> generated_tokens_in_loop;
    for (const auto &async_op : scope.async_ops) {
      generated_tokens_in_loop.insert(async_op.token);
    }

    std::set<int> hoistable_tokens;

    for (const auto &async_op : scope.async_ops) {
      std::set<int> loop_carried_waits;
      CollectLoopCarriedWaitsForToken(async_op.token, &loop_carried_waits);
      for (int token : loop_carried_waits) {
        AddHoistableToken(token, generated_tokens_in_loop, &hoistable_tokens);
      }
      for (const auto &read : async_op.reads) {
        AccessRecord covered_read = CoverAccessForLoopHoist(read, scope);
        CollectHoistableWaitsForReadAccess(
            covered_read, generated_tokens_in_loop, &hoistable_tokens);
      }
      for (const auto &write : async_op.writes) {
        AccessRecord covered_write = CoverAccessForLoopHoist(write, scope);
        CollectHoistableWaitsForWriteAccess(
            covered_write, generated_tokens_in_loop, &hoistable_tokens);
      }
    }

    for (const auto &sync_access : scope.sync_accesses) {
      std::set<int> loop_carried_waits;
      CollectLoopCarriedWaitsForSyncStmt(sync_access.stmt, &loop_carried_waits);
      for (int token : loop_carried_waits) {
        AddHoistableToken(token, generated_tokens_in_loop, &hoistable_tokens);
      }
      for (const auto &read : sync_access.reads) {
        AccessRecord covered_read = CoverAccessForLoopHoist(read, scope);
        CollectHoistableWaitsForReadAccess(
            covered_read, generated_tokens_in_loop, &hoistable_tokens);
      }
      for (const auto &write : sync_access.writes) {
        AccessRecord covered_write = CoverAccessForLoopHoist(write, scope);
        CollectHoistableWaitsForWriteAccess(
            covered_write, generated_tokens_in_loop, &hoistable_tokens);
      }
    }

    return hoistable_tokens;
  }

  // Analyzes a read operation on a buffer region.
  // Checks for dependencies with pending writes (RAW) and inserts waits if
  // necessary. Records the read access for future dependency checks.
  void token_process_read_buffer(const BufferRegion &buffer_region,
                                 Array<Stmt> &stmts, int curr_token_id,
                                 bool is_async_stmt = true,
                                 bool is_log_buffer = true) {
    Buffer src_buffer = buffer_region->buffer;
    Region src_region = buffer_region->region;
    auto src = Array<ObjectRef>{src_buffer, src_region};
    // Tracks whether a token has already been waited on within the current loop
    // level or in any of the scopes recorded in loop_scopes .
    std::unordered_set<int> waited_tokens;

    // Check if the current read buffer has dependencies with existing write
    // buffers. If yes, we need to wait for the write to finish before reading.
    for (const Array<ObjectRef> &buf : write_buffers) {
      if (is_async_stmt && write_buffer_token_map[buf] == curr_token_id) {
        continue;
      }
      Buffer buf_buffer = Downcast<Buffer>(buf[0]);
      Region buf_region = Downcast<Region>(buf[1]);
      if (src_buffer.same_as(buf_buffer) &&
          RegionIntersect(src_region, buf_region)) {
        int token = write_buffer_token_map[buf];
        if (waited_tokens.count(token) == 0) {
          maybe_process_wait_token_and_barrier_wait(stmts, token);
          waited_tokens.insert(token);
        }
      }
    }

    // After processing the dependencies with existing buffers, we can add the
    // current read buffer to the list.
    if (is_async_stmt && is_log_buffer) {
      read_buffers.push_back(src);
      read_buffer_token_map.Set(src, curr_token_id);
    }
  }

  // Analyzes a write operation on a buffer region.
  // Checks for dependencies with pending reads (WAR) and writes (WAW).
  // Inserts waits if necessary and records the write access.
  void token_process_write_buffer(const BufferRegion &buffer_region,
                                  Array<Stmt> &stmts, int curr_token_id,
                                  bool is_async_stmt = true,
                                  bool is_log_buffer = true) {
    Buffer dst_buffer = buffer_region->buffer;
    Region dst_region = buffer_region->region;
    auto dst = Array<ObjectRef>{dst_buffer, dst_region};
    std::unordered_set<int> waited_tokens;

    // Check if the current write buffer has dependencies with existing read
    // buffers. If yes, we need to wait for the read to finish before writing.
    for (const Array<ObjectRef> &buf : read_buffers) {
      if (is_async_stmt && read_buffer_token_map[buf] == curr_token_id) {
        continue;
      }
      Buffer buf_buffer = Downcast<Buffer>(buf[0]);
      Region buf_region = Downcast<Region>(buf[1]);
      if (dst_buffer.same_as(buf_buffer) &&
          RegionIntersect(dst_region, buf_region)) {
        int token = read_buffer_token_map[buf];
        if (waited_tokens.count(token) == 0) {
          maybe_process_wait_token_and_barrier_wait(stmts, token);
          waited_tokens.insert(token);
        }
      }
    }
    // We also need to check the dependencies with existing write buffers. If
    // yes, we need to wait for the write to finish before writing.
    for (const Array<ObjectRef> &buf : write_buffers) {
      if (is_async_stmt && write_buffer_token_map[buf] == curr_token_id) {
        continue;
      }
      Buffer buf_buffer = Downcast<Buffer>(buf[0]);
      Region buf_region = Downcast<Region>(buf[1]);
      if (dst_buffer.same_as(buf_buffer) &&
          RegionIntersect(dst_region, buf_region)) {
        int token = write_buffer_token_map[buf];
        if (waited_tokens.count(token) == 0) {
          maybe_process_wait_token_and_barrier_wait(stmts, token);
          waited_tokens.insert(token);
        }
      }
    }

    // After processing the dependencies with existing buffers, we can add the
    // current write buffer to the list.
    if (is_async_stmt && is_log_buffer) {
      write_buffers.push_back(dst);
      write_buffer_token_map.Set(dst, curr_token_id);
    }
  }

  // append the token_id to the end of the call arguments, and wrap it with
  // Evaluate.
  void curr_stmt_with_token_id(const CallNode *call, Array<Stmt> &stmts,
                               int token_id) {
    Array<PrimExpr> new_args = call->args;
    new_args.push_back(Call(DataType::Handle(), sync_token_id(),
                            {IntImm(DataType::Int(32), token_id)}));
    stmts.push_back(Evaluate(Call(call->dtype, call->op, new_args)));
    available_tokens_.insert(token_id);
  }

  // Computes the global participant core mask for a broadcast operation.
  // tl.broadcast_ uses args = [src_region, dst_region, direction, mask,
  // src_offset_byte, optional src_core, optional sync_token_id]. The optional
  // src_core is immediately before sync_token_id when the token is present.
  // The broadcast mask is direction-local; barriers still use global core ids.
  PrimExpr BroadcastParticipantMask(const CallNode *call) {
    ICHECK_GE(call->args.size(), static_cast<size_t>(kBroadcastArgCount))
        << "broadcast_() call is missing its fixed argument prefix.";
    int total_cores = mesh_nrow_ * mesh_ncol_;
    ICHECK_LE(total_cores, 64)
        << "tl.broadcast_ barrier mask currently supports at most 64 cores";
    int direction = -1;
    if (const auto *direction_imm =
            call->args[kBroadcastArgDirection].as<IntImmNode>()) {
      direction = static_cast<int>(direction_imm->value);
    }
    ICHECK(direction == 0 || direction == 1)
        << "tl.broadcast_ barrier mask expansion only supports horizontal or "
           "vertical leaf broadcasts";

    PrimExpr src_core = GetBroadcastBarrierSrcCore(call);
    PrimExpr local_mask = call->args[kBroadcastArgMask];
    PrimExpr write_mask =
        ExpandBroadcastLocalMask(local_mask, direction, src_core);
    PrimExpr read_mask = CoreBitMask(src_core);
    PrimExpr participant_mask = AsI64(read_mask) | AsI64(write_mask);
    return analyzer_ ? analyzer_->Simplify(participant_mask) : participant_mask;
  }

  BarrierMaskInfo BroadcastBarrierMaskInfo(const CallNode *call) {
    BarrierMaskInfo info;
    info.expr = AsI64(BroadcastParticipantMask(call));
    if (info.expr.as<IntImmNode>()) {
      return info;
    }

    int direction = -1;
    if (const auto *direction_imm =
            call->args[kBroadcastArgDirection].as<IntImmNode>()) {
      direction = static_cast<int>(direction_imm->value);
    }
    ICHECK(direction == 0 || direction == 1)
        << "tl.broadcast_ barrier candidate generation only supports "
           "horizontal or vertical leaf broadcasts";

    std::vector<int64_t> enumerated = EnumerateMaskCandidates(
        info.expr, direction, mesh_nrow_, mesh_ncol_, analyzer_);
    AppendCandidates(&info.candidates, enumerated);

    ICHECK(!info.candidates.empty())
        << "Could not derive static candidate masks for dynamic "
           "tl.broadcast_ barrier mask";
    return info;
  }

  BarrierMaskInfo RecordBroadcastBarrier(const CallNode *call,
                                         int curr_token_id) {
    BarrierMaskInfo info = BroadcastBarrierMaskInfo(call);
    token_to_barrier_mask_[curr_token_id] = info;
    return info;
  }

  void process_barrier_wait(Array<Stmt> &stmts,
                            const BarrierMaskInfo &participant_mask) {
    stmts.push_back(Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                                  MakeBarrierArgs(participant_mask))));
  }

  // Extracts all buffer read and write accesses from a primitive expression
  // and processes their dependencies to inject necessary synchronization
  // tokens.
  void token_process_prim_expr(const PrimExpr &expr, Array<Stmt> &stmts) {
    auto buf_load_collector =
        BufferAccessCollector(buffer_data_to_buffer_, loop_domains_);
    buf_load_collector(expr);
    Array<BufferRegion> read_regions = buf_load_collector.GetReads();
    for (const auto &read_region : read_regions) {
      token_process_region_bounds(read_region, stmts);
      token_process_read_buffer(read_region, stmts, -1, false);
    }
  }

  void token_process_region_bounds(const BufferRegion &buffer_region,
                                   Array<Stmt> &stmts) {
    for (const Range &range : buffer_region->region) {
      token_process_prim_expr(range->min, stmts);
      token_process_prim_expr(range->extent, stmts);
    }
  }

  void
  token_process_async_call_arguments(const CallNode *call,
                                     std::initializer_list<size_t> region_args,
                                     Array<Stmt> &stmts) {
    size_t arg_count = call->args.size();
    if (arg_count != 0 && IsSyncTokenExpr(call->args.back())) {
      --arg_count;
    }
    for (size_t i = 0; i < arg_count; ++i) {
      if (std::find(region_args.begin(), region_args.end(), i) !=
          region_args.end()) {
        token_process_region_bounds(NormalizeToBufferRegion(call->args[i]),
                                    stmts);
      } else {
        token_process_prim_expr(call->args[i], stmts);
      }
    }
  }

  Stmt VisitStmt_(const AttrStmtNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->value, stmts);

    PrimExpr old_kernel_core_id = current_kernel_core_id_;
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "blockIdx.x") {
        current_kernel_core_id_ = iv->var;
      }
    }
    stmts.push_back(StmtMutator::VisitStmt_(op));
    current_kernel_core_id_ = old_kernel_core_id;
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const LetStmtNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->value, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const WhileNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->condition, stmts);

    LoopAsyncCollector collector(buffer_data_to_buffer_);
    collector(op->body);

    LoopScope scope;
    scope.async_ops = collector.async_ops;
    scope.sync_accesses = collector.sync_accesses;
    scope.stmt_order = collector.stmt_order;
    std::vector<const EvaluateNode *> owned_pre_assignments;
    for (auto &async_op : scope.async_ops) {
      // Pre-assign a stable token id for each async site in this loop.
      // This lets the body rewriter attach the same token id every iteration,
      // enabling consistent loop-carried dependency reasoning.
      int token = PreAssignLoopToken(async_op, &owned_pre_assignments);
      async_op.token = token;

      // Keep a back-reference from token -> call for special handling after we
      // finish rewriting the loop (e.g. broadcast barrier initialization).
      const CallNode *call = async_op.call;
      scope.token_to_call[token] = call;

      if (call && call->op.same_as(broadcast_())) {
        RecordBroadcastBarrier(call, token);
      }
    }

    AnalyzeWhileLoopCarriedDependencies(&scope);

    InjectLoopEntryNullTokens(scope, stmts);

    // Push this loop scope so nested visitors can consult it when analyzing
    // read/write accesses inside the loop body.
    loop_scopes_.push_back(scope);
    std::set<int> hoisted_wait_tokens =
        AnalyzeHoistableWaitsForLoop(loop_scopes_.back());
    for (int token : hoisted_wait_tokens) {
      process_wait_token_and_barrier_wait(stmts, token);
    }
    loop_hoisted_wait_tokens_stack_.push_back(hoisted_wait_tokens);

    Stmt loop_stmt = StmtMutator::VisitStmt_(op);

    scope = loop_scopes_.back();
    loop_scopes_.pop_back();
    loop_hoisted_wait_tokens_stack_.pop_back();
    ClearOwnedPreAssignedTokens(owned_pre_assignments);

    stmts.push_back(loop_stmt);
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AllocateNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->condition, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const BufferRealizeNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->condition, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AssertStmtNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->condition, stmts);
    token_process_prim_expr(op->message, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->predicate, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);

    // For a buffer store statement, we need to check the dependencies for the
    // buffer to be stored. For example, in the statement A[i] = B[j] + C[k], we
    // need to check the dependencies for the buffer A.
    Buffer store_buffer = op->buffer;
    Array<PrimExpr> indices = op->indices;
    // convert indices to region
    Array<Range> region;
    for (const auto &index : indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    for (const auto &index : indices) {
      token_process_prim_expr(index, stmts);
    }
    auto store_region = BufferRegion(
        store_buffer, CoverRegionWithLoopDomains(region, loop_domains_));
    token_process_write_buffer(store_region, stmts, -1, false);

    // For a store statement, we also need to check the read dependencies for
    // the value to be stored. For example, in the statement A[i] = B[j] + C[k],
    // we need to check the read dependencies for the buffers B and C.
    token_process_prim_expr(op->value, stmts);

    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  // Handles specific async instructions (dma_copy, mma_sunmmio, broadcast).
  // Assigns tokens/barriers and registers them for dependency tracking.
  Stmt VisitStmt_(const EvaluateNode *op) {
    const CallNode *call = op->value.as<CallNode>();
    if (call) {
      if (call->op.same_as(dma_copy())) {
        Array<Stmt> stmts;
        int curr_token_id;
        if (pre_assigned_tokens_.count(op)) {
          curr_token_id = pre_assigned_tokens_[op];
        } else {
          curr_token_id = GetNextTokenId();
        }

        InjectLoopCarriedWaitsForToken(stmts, curr_token_id);
        token_process_async_call_arguments(call, {0, 1}, stmts);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[0]), stmts,
                                  curr_token_id);
        token_process_write_buffer(NormalizeToBufferRegion(call->args[1]),
                                   stmts, curr_token_id);

        curr_stmt_with_token_id(call, stmts, curr_token_id);

        return SeqStmt::Flatten(stmts);
      } else if (call->op.same_as(sunmmio_layout_transform())) {
        // Same dependency shape as dma_copy: reads args[0], writes args[1].
        Array<Stmt> stmts;
        int curr_token_id;
        if (pre_assigned_tokens_.count(op)) {
          curr_token_id = pre_assigned_tokens_[op];
        } else {
          curr_token_id = GetNextTokenId();
        }

        InjectLoopCarriedWaitsForToken(stmts, curr_token_id);
        token_process_async_call_arguments(call, {0, 1}, stmts);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[0]), stmts,
                                  curr_token_id);
        token_process_write_buffer(NormalizeToBufferRegion(call->args[1]),
                                   stmts, curr_token_id);

        curr_stmt_with_token_id(call, stmts, curr_token_id);

        return SeqStmt::Flatten(stmts);
      } else if (call->op.same_as(mma_sunmmio())) {
        Array<Stmt> stmts;
        int curr_token_id;
        if (pre_assigned_tokens_.count(op)) {
          curr_token_id = pre_assigned_tokens_[op];
        } else {
          curr_token_id = GetNextTokenId();
        }

        InjectLoopCarriedWaitsForToken(stmts, curr_token_id);
        token_process_async_call_arguments(call, {0, 1, 2}, stmts);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[0]), stmts,
                                  curr_token_id);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[1]), stmts,
                                  curr_token_id);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[2]), stmts,
                                  curr_token_id, true, false);
        token_process_write_buffer(NormalizeToBufferRegion(call->args[2]),
                                   stmts, curr_token_id);

        curr_stmt_with_token_id(call, stmts, curr_token_id);

        return SeqStmt::Flatten(stmts);
      } else if (call->op.same_as(broadcast_())) {
        Array<Stmt> stmts;
        int curr_token_id;
        if (pre_assigned_tokens_.count(op)) {
          curr_token_id = pre_assigned_tokens_[op];
        } else {
          curr_token_id = GetNextTokenId();
        }

        InjectLoopCarriedWaitsForToken(stmts, curr_token_id);
        token_process_async_call_arguments(call, {0, 1}, stmts);
        BarrierMaskInfo participant_mask =
            RecordBroadcastBarrier(call, curr_token_id);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[0]), stmts,
                                  curr_token_id);
        token_process_write_buffer(NormalizeToBufferRegion(call->args[1]),
                                   stmts, curr_token_id);

        process_barrier_wait(stmts, participant_mask);
        curr_stmt_with_token_id(call, stmts, curr_token_id);

        return SeqStmt::Flatten(stmts);
      }
    }

    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->value, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  // Handles control flow splitting (IfThenElse).
  // We need to track buffer states independently for then/else branches and
  // then merge them.
  Stmt VisitStmt_(const IfThenElseNode *op) {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, op);
    token_process_prim_expr(op->condition, stmts);
    PrimExpr condition = this->VisitExpr(op->condition);

    Stmt then_case;
    ffi::Optional<Stmt> else_case = std::nullopt;
    if (op->else_case) {
      Array<Array<ObjectRef>> read_buffers_before(read_buffers);
      Array<Array<ObjectRef>> write_buffers_before(write_buffers);
      Map<Array<ObjectRef>, int> read_buffer_token_map_before(
          read_buffer_token_map);
      Map<Array<ObjectRef>, int> write_buffer_token_map_before(
          write_buffer_token_map);

      then_case = this->VisitStmt(op->then_case);

      Array<Array<ObjectRef>> read_buffers_after_then(read_buffers);
      Array<Array<ObjectRef>> write_buffers_after_then(write_buffers);
      Map<Array<ObjectRef>, int> read_buffer_token_map_after_then(
          read_buffer_token_map);
      Map<Array<ObjectRef>, int> write_buffer_token_map_after_then(
          write_buffer_token_map);

      read_buffers = read_buffers_before;
      write_buffers = write_buffers_before;
      read_buffer_token_map = read_buffer_token_map_before;
      write_buffer_token_map = write_buffer_token_map_before;

      else_case = this->VisitStmt(op->else_case.value());

      for (auto i = read_buffers_before.size(); i < read_buffers.size(); i++) {
        auto buf = read_buffers[i];
        read_buffers_after_then.push_back(buf);
        read_buffer_token_map_after_then.Set(buf, read_buffer_token_map[buf]);
      }
      read_buffers = read_buffers_after_then;
      read_buffer_token_map = read_buffer_token_map_after_then;
      for (auto i = write_buffers_before.size(); i < write_buffers.size();
           i++) {
        auto buf = write_buffers[i];
        write_buffers_after_then.push_back(buf);
        write_buffer_token_map_after_then.Set(buf, write_buffer_token_map[buf]);
      }
      write_buffers = write_buffers_after_then;
      write_buffer_token_map = write_buffer_token_map_after_then;
    } else {
      then_case = this->VisitStmt(op->then_case);
    }

    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      stmts.push_back(ffi::GetRef<Stmt>(op));
    } else {
      auto n = CopyOnWrite(op);
      n->condition = std::move(condition);
      n->then_case = std::move(then_case);
      n->else_case = std::move(else_case);
      stmts.push_back(Stmt(n));
    }
    return SeqStmt::Flatten(stmts);
  }

  // Handles loops.
  // We pre-assign tokens to async writes in the loop to handle loop-carried
  // dependencies.
  Stmt VisitStmt_(const ForNode *loop) final {
    Array<Stmt> stmts;
    InjectLoopCarriedWaitsForSyncStmt(stmts, loop);
    token_process_prim_expr(loop->min, stmts);
    token_process_prim_expr(loop->extent, stmts);

    LoopAsyncCollector collector(buffer_data_to_buffer_);
    collector(loop->body);

    LoopScope scope;
    scope.loop_var = loop->loop_var;
    scope.loop_min = loop->min;
    scope.loop_extent = loop->extent;
    scope.async_ops = collector.async_ops;
    scope.sync_accesses = collector.sync_accesses;
    scope.stmt_order = collector.stmt_order;

    std::vector<const EvaluateNode *> owned_pre_assignments;
    for (auto &async_op : scope.async_ops) {
      int token = PreAssignLoopToken(async_op, &owned_pre_assignments);
      async_op.token = token;

      const CallNode *call = async_op.call;
      scope.token_to_call[token] = call;

      // check if it is a broadcast
      if (call && call->op.same_as(broadcast_())) {
        RecordBroadcastBarrier(call, token);
      }
    }

    AnalyzeLoopCarriedDependencies(&scope);

    InjectLoopEntryNullTokens(scope, stmts);

    loop_scopes_.push_back(scope);
    std::set<int> hoisted_wait_tokens =
        AnalyzeHoistableWaitsForLoop(loop_scopes_.back());
    for (int token : hoisted_wait_tokens) {
      process_wait_token_and_barrier_wait(stmts, token);
    }
    loop_hoisted_wait_tokens_stack_.push_back(hoisted_wait_tokens);

    ffi::Map<Var, arith::IntSet> old_loop_domains = loop_domains_;
    if (CanUseLoopDomainForRegionCover(loop->min) &&
        CanUseLoopDomainForRegionCover(loop->extent)) {
      loop_domains_.Set(loop->loop_var,
                        arith::IntSet::FromRange(
                            Range::FromMinExtent(loop->min, loop->extent)));
    }
    Stmt loop_stmt = StmtMutator::VisitStmt_(loop);
    loop_domains_ = std::move(old_loop_domains);

    scope = loop_scopes_.back();
    loop_scopes_.pop_back();
    loop_hoisted_wait_tokens_stack_.pop_back();
    ClearOwnedPreAssignedTokens(owned_pre_assignments);

    stmts.push_back(loop_stmt);

    if (const auto *realize = loop->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
    }
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return std::move(block);
  }

private:
  int GetNextTokenId() { return token_count++; }

  int token_count;
  int mesh_nrow_;
  int mesh_ncol_;
  arith::Analyzer *analyzer_;
  bool insert_barrier_wait_after_wait_token_;

  Array<Array<ObjectRef>> read_buffers;
  Array<Array<ObjectRef>> write_buffers;
  Map<Array<ObjectRef>, int> read_buffer_token_map;
  Map<Array<ObjectRef>, int> write_buffer_token_map;
  TokenBarrierMap token_to_barrier_mask_;
  std::set<int> available_tokens_;

  Map<Var, Buffer> buffer_data_to_buffer_;
  PrimExpr current_kernel_core_id_;
  ffi::Map<Var, arith::IntSet> loop_domains_;
  std::vector<LoopScope> loop_scopes_;
  std::vector<std::set<int>> loop_hoisted_wait_tokens_stack_;
  std::map<const EvaluateNode *, int> pre_assigned_tokens_;
};

// Rewriter to repair loop-local token reuse after InjectSyncRewriter has
// attached concrete sync_token_id/wait_token calls. If a loop body generates a
// token but contains no wait for that token anywhere in the body, the next
// iteration would overwrite the previous iteration's token. Insert a wait
// before each generation site and seed the first iteration with a null token.
class LoopMissingTokenWaitRewriter : public StmtMutator {
public:
  Stmt operator()(Stmt body) { return this->VisitStmt(body); }

private:
  static std::optional<int> TryGetTokenId(const CallNode *call) {
    if (!call || call->args.empty()) {
      return std::nullopt;
    }
    if (const auto *imm = call->args[0].as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    return std::nullopt;
  }

  static Stmt MakeWaitTokenStmt(int token_id) {
    return Evaluate(Call(DataType::Handle(), wait_token(),
                         {IntImm(DataType::Int(32), token_id)}));
  }

  static Stmt MakeSyncNullTokenStmt(int token_id) {
    return Evaluate(Call(DataType::Handle(), sync_null_token(),
                         {IntImm(DataType::Int(32), token_id)}));
  }

  class LoopTokenUseCollector : public StmtExprVisitor {
  public:
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(sync_token_id())) {
        if (std::optional<int> token_id = TryGetTokenId(op)) {
          generated_tokens.insert(*token_id);
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (const CallNode *call = op->value.as<CallNode>()) {
        if (call->op.same_as(wait_token())) {
          if (std::optional<int> token_id = TryGetTokenId(call)) {
            waited_tokens.insert(*token_id);
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    std::set<int> generated_tokens;
    std::set<int> waited_tokens;
  };

  class GeneratedTokenCollector : public StmtExprVisitor {
  public:
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(sync_token_id())) {
        if (std::optional<int> token_id = TryGetTokenId(op)) {
          tokens.insert(*token_id);
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const ForNode *op) final {}
    void VisitStmt_(const WhileNode *op) final {}

    std::set<int> tokens;
  };

  static void PushFlatten(Array<Stmt> *stmts, const Stmt &stmt) {
    if (!stmt.defined()) {
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) {
        PushFlatten(stmts, child);
      }
      return;
    }
    stmts->push_back(stmt);
  }

  std::set<int> GetMissingWaitTokens(const Stmt &body) const {
    LoopTokenUseCollector collector;
    collector(body);

    std::set<int> missing_tokens;
    for (int token_id : collector.generated_tokens) {
      if (collector.waited_tokens.count(token_id) == 0) {
        missing_tokens.insert(token_id);
      }
    }
    return missing_tokens;
  }

  Stmt PrependSyncNullTokens(const Stmt &loop_stmt,
                             const std::set<int> &tokens) const {
    if (tokens.empty()) {
      return loop_stmt;
    }
    Array<Stmt> stmts;
    for (int token_id : tokens) {
      stmts.push_back(MakeSyncNullTokenStmt(token_id));
    }
    stmts.push_back(loop_stmt);
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> out;
    for (const Stmt &stmt : op->seq) {
      PushFlatten(&out, VisitStmt(stmt));
    }
    return SeqStmt::Flatten(out);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = ffi::GetRef<Stmt>(op);
    GeneratedTokenCollector collector;
    collector(stmt);

    Array<Stmt> stmts;
    for (int token_id : collector.tokens) {
      if (tokens_to_wait_before_.count(token_id) != 0) {
        stmts.push_back(MakeWaitTokenStmt(token_id));
      }
    }
    stmts.push_back(stmt);
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    Stmt body = StmtMutator::VisitStmt(op->body);
    std::set<int> missing_tokens = GetMissingWaitTokens(body);
    if (!missing_tokens.empty()) {
      std::set<int> old_tokens_to_wait_before = tokens_to_wait_before_;
      tokens_to_wait_before_ = missing_tokens;
      body = VisitStmt(body);
      tokens_to_wait_before_ = std::move(old_tokens_to_wait_before);
    }

    Stmt loop_stmt =
        For(op->loop_var, op->min, op->extent, op->kind, body,
            op->thread_binding, op->annotations, std::nullopt, op->span);
    return PrependSyncNullTokens(loop_stmt, missing_tokens);
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    Stmt body = StmtMutator::VisitStmt(op->body);
    std::set<int> missing_tokens = GetMissingWaitTokens(body);
    if (!missing_tokens.empty()) {
      std::set<int> old_tokens_to_wait_before = tokens_to_wait_before_;
      tokens_to_wait_before_ = missing_tokens;
      body = VisitStmt(body);
      tokens_to_wait_before_ = std::move(old_tokens_to_wait_before);
    }

    Stmt loop_stmt = While(op->condition, body);
    return PrependSyncNullTokens(loop_stmt, missing_tokens);
  }

  std::set<int> tokens_to_wait_before_;
};

// Rewriter to inject final synchronization waits before the device function
// returns. This ensures all pending asynchronous operations are completed
// before the device kernel finishes, handling both explicit returns and
// implicit function exits.
class DeviceFuncWaitRewriter : public StmtMutator {
public:
  DeviceFuncWaitRewriter(TokenBarrierMap token_to_barrier_mask)
      : token_to_barrier_mask_(std::move(token_to_barrier_mask)) {}

  Stmt operator()(Stmt body) { return this->VisitStmt(body); }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      bool is_outermost_thread_extent = thread_extent_depth_ == 0;
      ++thread_extent_depth_;
      Stmt body = StmtMutator::VisitStmt(op->body);
      --thread_extent_depth_;

      if (!is_outermost_thread_extent) {
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }

      DevicePendingTokenCollector collector;
      collector(body);

      if (collector.pending_tokens.empty()) {
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }

      Array<Stmt> stmts;
      if (const auto *seq = body.as<SeqStmtNode>()) {
        stmts = seq->seq;
      } else {
        stmts.push_back(body);
      }

      std::vector<int> tokens(collector.pending_tokens.begin(),
                              collector.pending_tokens.end());
      std::sort(tokens.begin(), tokens.end());

      for (int token_id : tokens) {
        stmts.push_back(Evaluate(Call(DataType::Handle(), wait_token(),
                                      {IntImm(DataType::Int(32), token_id)})));
        auto barrier_it = token_to_barrier_mask_.find(token_id);
        if (barrier_it != token_to_barrier_mask_.end()) {
          stmts.push_back(
              Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                            MakeBarrierArgs(barrier_it->second))));
        }
      }
      return AttrStmt(op->node, op->attr_key, op->value,
                      SeqStmt::Flatten(stmts));
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    return StmtMutator::VisitStmt_(op);
  }

private:
  TokenBarrierMap token_to_barrier_mask_;
  int thread_extent_depth_{0};

  // Helper to collect token IDs that are still pending at device-function exit.
  class DevicePendingTokenCollector : public StmtExprVisitor {
  public:
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(sync_token_id())) {
        int token_id = op->args[0].as<IntImm>().value()->value;
        pending_tokens.insert(token_id);
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (const CallNode *call = op->value.as<CallNode>()) {
        if (call->op.same_as(wait_token())) {
          if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
            pending_tokens.erase(call->args[0].as<IntImmNode>()->value);
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      auto pending_before = pending_tokens;

      VisitStmt(op->then_case);
      auto then_pending = pending_tokens;

      pending_tokens = pending_before;
      if (op->else_case.defined()) {
        VisitStmt(op->else_case.value());
      }

      pending_tokens.insert(then_pending.begin(), then_pending.end());
    }

    void VisitStmt_(const ForNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const WhileNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const SeqStmtNode *op) final {
      for (const Stmt &stmt : op->seq) {
        VisitStmt(stmt);
      }
    }

    std::set<int> pending_tokens;
  };
};

// Collector to identify all sync tokens generated within a statement or
// expression. This is primarily used for tracking resources that may need
// subsequent synchronizations.
class AsyncResourceCollector : public StmtExprVisitor {
public:
  std::set<int> generated_tokens;

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(sync_token_id()) || op->op.same_as(sync_null_token())) {
      if (!op->args.empty() && op->args[0].as<IntImmNode>()) {
        int token_id = op->args[0].as<IntImmNode>()->value;
        generated_tokens.insert(token_id);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

// Analyzer to track which tokens are currently pending (i.e., generated but not
// yet waited on) within a specific execution scope. Used to determine if
// additional waits are required. Particularly note the following scenario:
// dependent tokens within a loop may lack a corresponding wait after the final
// iteration.
class PendingAnalyzer : public StmtExprVisitor {
public:
  std::set<int> pending_tokens;

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(sync_token_id()) || op->op.same_as(sync_null_token())) {
      if (!op->args.empty() && op->args[0].as<IntImmNode>()) {
        pending_tokens.insert(op->args[0].as<IntImmNode>()->value);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(wait_token())) {
        if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
          pending_tokens.erase(call->args[0].as<IntImmNode>()->value);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    auto pending_tokens_before = pending_tokens;

    VisitStmt(op->then_case);
    auto then_pending_tokens = pending_tokens;

    pending_tokens = pending_tokens_before;

    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }

    pending_tokens.insert(then_pending_tokens.begin(),
                          then_pending_tokens.end());
  }

  void VisitStmt_(const ForNode *op) final { VisitStmt(op->body); }

  void VisitStmt_(const WhileNode *op) final { VisitStmt(op->body); }

  void VisitStmt_(const SeqStmtNode *op) final {
    for (auto stmt : op->seq) {
      VisitStmt(stmt);
    }
  }
};

// Collector to identify all sync tokens that are explicitly waited on within a
// given statement.
class ResolvedResourceCollector : public StmtExprVisitor {
public:
  std::set<int> resolved_tokens;

  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(wait_token())) {
        if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
          resolved_tokens.insert(call->args[0].as<IntImmNode>()->value);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

// Optimization pass to remove redundant token waits. Barrier arrive-and-wait is
// kept as the paired synchronization event for the token wait that introduced
// it; it is not used to infer token completion by participant mask.
class EliminateRedundancyRewriter : public StmtMutator {
public:
  EliminateRedundancyRewriter(arith::Analyzer *analyzer = nullptr,
                              std::vector<int> parent_token_ids = {},
                              TokenBarrierMap token_to_barrier_mask = {})
      : analyzer_(analyzer), parent_token_ids_(std::move(parent_token_ids)),
        token_to_barrier_mask_(std::move(token_to_barrier_mask)) {
    current_token_ids_ = {};
  }

  std::vector<int> get_current_token_ids() const { return current_token_ids_; }

private:
  std::vector<int> get_all_token_ids() const {
    std::vector<int> all_token_ids = parent_token_ids_;
    all_token_ids.insert(all_token_ids.end(), current_token_ids_.begin(),
                         current_token_ids_.end());
    return all_token_ids;
  }

  bool IsTokenResolved(int token_id) const {
    return std::find(parent_token_ids_.begin(), parent_token_ids_.end(),
                     token_id) != parent_token_ids_.end() ||
           std::find(current_token_ids_.begin(), current_token_ids_.end(),
                     token_id) != current_token_ids_.end();
  }

  void MarkTokenPending(int token_id) {
    current_token_ids_.erase(std::remove(current_token_ids_.begin(),
                                         current_token_ids_.end(), token_id),
                             current_token_ids_.end());
  }

  void MarkGeneratedAsyncTokensPending(const Stmt &stmt) {
    class Collector final : public StmtExprVisitor {
    public:
      void VisitStmt_(const AttrStmtNode *op) final {}
      void VisitStmt_(const ForNode *op) final {}
      void VisitStmt_(const IfThenElseNode *op) final {}
      void VisitStmt_(const WhileNode *op) final {}

      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(sync_token_id()) && !op->args.empty()) {
          if (const auto *imm = op->args[0].as<IntImmNode>()) {
            token_ids.push_back(imm->value);
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      std::vector<int> token_ids;
    };

    Collector collector;
    collector(stmt);
    for (int token_id : collector.token_ids) {
      MarkTokenPending(token_id);
    }
  }

  bool IsBarrierWaitForToken(const Stmt &stmt, int token_id) const {
    auto it = token_to_barrier_mask_.find(token_id);
    if (it == token_to_barrier_mask_.end()) {
      return false;
    }
    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return false;
    }
    const auto *call = eval->value.as<CallNode>();
    return BarrierCallMatchesInfo(call, it->second);
  }

  void MarkTokenResolved(int token_id) {
    if (std::find(current_token_ids_.begin(), current_token_ids_.end(),
                  token_id) == current_token_ids_.end()) {
      current_token_ids_.push_back(token_id);
    }
  }

  // Propagates the resolved token states from a block (e.g., loop body or if
  // branch) to the current scope, marking them as handled to avoid redundant
  // waits.
  void PropagateResolvedStates(const Stmt &block,
                               bool guaranteed_to_execute = false) {
    AsyncResourceCollector collector;
    collector(block);

    // Analyze which of the collected resources are still pending after the
    // block finishes. A resource is considered "pending" if there exists a
    // path in the block that may still require a corresponding wait later.
    PendingAnalyzer pending_analyzer;
    pending_analyzer(block);

    for (int token_id : collector.generated_tokens) {
      if (pending_analyzer.pending_tokens.count(token_id) == 0) {
        MarkTokenResolved(token_id);
      }
    }

    // If the block is guaranteed to execute, any explicit waits within the
    // block that are not pending at the end are also resolved for the parent
    // scope.
    if (guaranteed_to_execute) {
      ResolvedResourceCollector resolved_collector;
      resolved_collector(block);

      for (int token_id : resolved_collector.resolved_tokens) {
        if (pending_analyzer.pending_tokens.count(token_id) == 0) {
          MarkTokenResolved(token_id);
        }
      }
    }
  }

  bool MatchWaitTokenStmt(const Stmt &stmt, int *token_id) const {
    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return false;
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call || !call->op.same_as(wait_token()) || call->args.size() != 1) {
      return false;
    }
    const auto *imm = call->args[0].as<IntImmNode>();
    if (!imm) {
      return false;
    }
    *token_id = imm->value;
    return true;
  }

  bool MatchBarrierWaitStmt(const Stmt &stmt, BarrierMaskInfo *mask) const {
    const auto *eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return false;
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call || !call->op.same_as(barrier_arrive_and_wait()) ||
        call->args.empty()) {
      return false;
    }
    *mask = BarrierMaskInfoFromArgs(call->args);
    return true;
  }

  void PushNonRedundantStmt(Array<Stmt> *out, const Stmt &stmt) const {
    if (!stmt.defined()) {
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) {
        PushNonRedundantStmt(out, child);
      }
      return;
    }
    if (!out->empty()) {
      BarrierMaskInfo prev_mask;
      BarrierMaskInfo curr_mask;
      if (MatchBarrierWaitStmt((*out)[out->size() - 1], &prev_mask) &&
          MatchBarrierWaitStmt(stmt, &curr_mask) &&
          SameBarrierMaskInfo(prev_mask, curr_mask)) {
        return;
      }
    }
    out->push_back(stmt);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> out;
    out.reserve(op->seq.size());
    for (int i = 0, n = static_cast<int>(op->seq.size()); i < n; ++i) {
      int token_id = -1;
      if (MatchWaitTokenStmt(op->seq[i], &token_id)) {
        if (IsTokenResolved(token_id)) {
          if (i + 1 < n && IsBarrierWaitForToken(op->seq[i + 1], token_id)) {
            ++i;
          }
          continue;
        }
        MarkTokenResolved(token_id);
      }
      Stmt rewritten = VisitStmt(op->seq[i]);
      MarkGeneratedAsyncTokensPending(rewritten);
      PushNonRedundantStmt(&out, rewritten);
    }
    return SeqStmt::Flatten(out);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    auto body_rewriter = EliminateRedundancyRewriter(
        analyzer_, get_all_token_ids(), token_to_barrier_mask_);
    Stmt body = body_rewriter(op->body);
    PropagateResolvedStates(body, /*guaranteed_to_execute=*/true);
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) {
    auto eliminate_sync_then_rewriter = EliminateRedundancyRewriter(
        analyzer_, get_all_token_ids(), token_to_barrier_mask_);
    auto then_case = eliminate_sync_then_rewriter(op->then_case);

    Stmt else_case;
    if (op->else_case.defined()) {
      auto eliminate_sync_else_rewriter = EliminateRedundancyRewriter(
          analyzer_, get_all_token_ids(), token_to_barrier_mask_);
      else_case = eliminate_sync_else_rewriter(op->else_case.value());

      std::vector<int> then_tokens =
          eliminate_sync_then_rewriter.get_current_token_ids();
      std::vector<int> else_tokens =
          eliminate_sync_else_rewriter.get_current_token_ids();
      for (int t_id : then_tokens) {
        if (std::find(else_tokens.begin(), else_tokens.end(), t_id) !=
            else_tokens.end()) {
          if (std::find(current_token_ids_.begin(), current_token_ids_.end(),
                        t_id) == current_token_ids_.end()) {
            current_token_ids_.push_back(t_id);
          }
        }
      }
    }

    auto new_stmt = IfThenElse(op->condition, then_case, else_case);
    PropagateResolvedStates(new_stmt);

    return new_stmt;
  }

  Stmt VisitStmt_(const ForNode *op) {
    auto eliminate_sync_loop_rewriter = EliminateRedundancyRewriter(
        analyzer_, get_all_token_ids(), token_to_barrier_mask_);
    auto body = eliminate_sync_loop_rewriter(op->body);

    bool is_guaranteed = false;
    if (analyzer_) {
      if (analyzer_->CanProveGreaterEqual(op->extent, 1)) {
        is_guaranteed = true;
      }
    } else if (auto extent = op->extent.as<IntImmNode>()) {
      if (extent->value > 0) {
        is_guaranteed = true;
      }
    }

    PropagateResolvedStates(ffi::GetRef<Stmt>(op), is_guaranteed);

    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations);
  }

  Stmt VisitStmt_(const WhileNode *op) {
    auto eliminate_sync_loop_rewriter = EliminateRedundancyRewriter(
        analyzer_, get_all_token_ids(), token_to_barrier_mask_);
    auto body = eliminate_sync_loop_rewriter(op->body);

    bool is_guaranteed = false;
    if (auto cond = op->condition.as<IntImmNode>()) {
      if (cond->value != 0) {
        is_guaranteed = true;
      }
    }

    PropagateResolvedStates(ffi::GetRef<Stmt>(op), is_guaranteed);

    return While(op->condition, body);
  }

private:
  arith::Analyzer *analyzer_;
  // Token IDs that are already known to be waited/synchronized in outer scopes
  std::vector<int> parent_token_ids_;
  // Token IDs that have been waited/synchronized along the current execution
  // path
  std::vector<int> current_token_ids_;
  TokenBarrierMap token_to_barrier_mask_;
};

// A reusable barrier represents one collective epoch, so two broadcasts with
// the same participant mask still require two arrivals.  Restore the launch
// invariant after wait elimination: every all-gather leaf is immediately
// preceded by its barrier.  This pass does not reorder scheduled commands.
class EnsureAllGatherLaunchBarrierRewriter : public StmtMutator {
public:
  explicit EnsureAllGatherLaunchBarrierRewriter(
      TokenBarrierMap token_to_barrier_mask)
      : token_to_barrier_mask_(std::move(token_to_barrier_mask)) {}

private:
  static const CallNode *GetEvaluateCall(const Stmt &stmt) {
    const auto *eval = stmt.as<EvaluateNode>();
    return eval ? eval->value.as<CallNode>() : nullptr;
  }

  static std::optional<int> GetGeneratedToken(const CallNode *call) {
    if (!call) {
      return std::nullopt;
    }
    for (const PrimExpr &arg : call->args) {
      const auto *nested = arg.as<CallNode>();
      if (nested && nested->op.same_as(sync_token_id()) &&
          !nested->args.empty()) {
        if (const auto *imm = nested->args[0].as<IntImmNode>()) {
          return static_cast<int>(imm->value);
        }
      }
    }
    return std::nullopt;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> out;
    for (const Stmt &old_stmt : op->seq) {
      Stmt stmt = VisitStmt(old_stmt);
      const CallNode *call = GetEvaluateCall(stmt);
      if (call && call->op.same_as(broadcast_()) &&
          !BroadcastCallHasSrcCore(call)) {
        std::optional<int> token = GetGeneratedToken(call);
        ICHECK(token.has_value())
            << "all-gather broadcast is missing its synchronization token";
        auto barrier_it = token_to_barrier_mask_.find(*token);
        ICHECK(barrier_it != token_to_barrier_mask_.end())
            << "all-gather token " << *token << " has no barrier mask";

        const CallNode *previous =
            out.empty() ? nullptr : GetEvaluateCall(out[out.size() - 1]);
        if (!BarrierCallMatchesInfo(previous, barrier_it->second)) {
          out.push_back(
              Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                            MakeBarrierArgs(barrier_it->second))));
        }
      }
      out.push_back(stmt);
    }
    return SeqStmt::Flatten(out);
  }

  TokenBarrierMap token_to_barrier_mask_;
};

// Hardware waits synchronize an engine, not an individual logical token.  A
// pipeline may schedule an operation from iteration N+1 before the existing
// wait for iteration N's token.  If both use the same engine, the wait would
// also consume the newly launched operation and destroy the intended overlap.
// Move an older token's wait before the first newer reuse of its engine.  For
// loop-carried tokens this means before the first same-engine launch in the
// new iteration.  Cross-core barriers remain attached to collective launches.
class MoveLoopCarriedEngineWaitsRewriter : public StmtMutator {
private:
  enum class AsyncEngine { kODMA0 = 0, kODMA1 = 1, kTensorCore = 2 };

  static const CallNode *GetEvaluateCall(const Stmt &stmt) {
    const auto *eval = stmt.as<EvaluateNode>();
    return eval ? eval->value.as<CallNode>() : nullptr;
  }

  static std::optional<int> GetTokenId(const CallNode *call,
                                       const Op &token_op) {
    if (!call) {
      return std::nullopt;
    }
    if (call->op.same_as(token_op) && !call->args.empty()) {
      if (const auto *imm = call->args[0].as<IntImmNode>()) {
        return static_cast<int>(imm->value);
      }
    }
    for (const PrimExpr &arg : call->args) {
      const auto *nested = arg.as<CallNode>();
      if (nested && nested->op.same_as(token_op) && !nested->args.empty()) {
        if (const auto *imm = nested->args[0].as<IntImmNode>()) {
          return static_cast<int>(imm->value);
        }
      }
    }
    return std::nullopt;
  }

  static AsyncEngine GetODMAEngine(const CallNode *call) {
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    if ((src->buffer.scope() == "shared.rsram" ||
         src->buffer.scope() == "local") &&
        dst->buffer.scope() == "shared.asram") {
      return AsyncEngine::kODMA1;
    }
    return AsyncEngine::kODMA0;
  }

  static std::optional<AsyncEngine> GetAsyncEngine(const CallNode *call) {
    if (!call) {
      return std::nullopt;
    }
    if (call->op.same_as(mma_sunmmio())) {
      return AsyncEngine::kTensorCore;
    }
    if (call->op.same_as(dma_copy()) ||
        call->op.same_as(sunmmio_layout_transform()) ||
        call->op.same_as(broadcast_())) {
      return GetODMAEngine(call);
    }
    return std::nullopt;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    loop_stack_.push_back(op);
    Stmt body = VisitStmt(op->body);
    loop_stack_.pop_back();
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    for (const Stmt &stmt : op->seq) {
      visited.push_back(VisitStmt(stmt));
    }
    Stmt body = SeqStmt::Flatten(visited);
    if (loop_stack_.empty()) {
      return body;
    }
    return RewriteSequence(body, loop_stack_.back());
  }

  static Stmt RewriteSequence(const Stmt &body, const ForNode *loop) {
    const auto *seq = body.as<SeqStmtNode>();
    if (!seq) {
      return body;
    }

    std::map<int, AsyncEngine> token_engine;
    std::map<int, int> generator_index;
    std::vector<int> engine_users[3];
    for (int i = 0, n = static_cast<int>(seq->seq.size()); i < n; ++i) {
      const CallNode *call = GetEvaluateCall(seq->seq[i]);
      std::optional<AsyncEngine> engine = GetAsyncEngine(call);
      if (!engine) {
        continue;
      }
      int engine_index = static_cast<int>(*engine);
      engine_users[engine_index].push_back(i);
      std::optional<int> token = GetTokenId(call, sync_token_id());
      if (token) {
        token_engine[*token] = *engine;
        generator_index[*token] = i;
      }
    }

    struct MovedWait {
      Stmt stmt;
      bool loop_carried;
    };
    std::map<int, std::vector<MovedWait>> moves;
    std::set<int> removed;
    for (int i = 0, n = static_cast<int>(seq->seq.size()); i < n; ++i) {
      std::optional<int> token =
          GetTokenId(GetEvaluateCall(seq->seq[i]), wait_token());
      if (!token || !token_engine.count(*token)) {
        continue;
      }
      int generator = generator_index[*token];
      bool loop_carried = i < generator;
      int target = i;
      for (int user : engine_users[static_cast<int>(token_engine[*token])]) {
        // For a carried token, no generator has run in this iteration yet, so
        // the wait must precede the first same-engine launch.  Otherwise move
        // it only across newer launches inserted between its generator and
        // wait; never move it before its own generator.
        if ((loop_carried || user > generator) && user < i) {
          target = user;
          break;
        }
      }
      if (target == i) {
        continue;
      }
      moves[target].push_back({seq->seq[i], loop_carried});
      removed.insert(i);
    }
    if (moves.empty()) {
      return body;
    }

    Array<Stmt> reordered;
    for (int i = 0, n = static_cast<int>(seq->seq.size()); i < n; ++i) {
      auto move_it = moves.find(i);
      if (move_it != moves.end()) {
        for (const MovedWait &wait : move_it->second) {
          if (wait.loop_carried) {
            // The carried token is initialized with sync_null_token.  NPU-IR
            // lowers a wait on the loop block argument as a real hardware
            // wait, so explicitly skip it on the first iteration.
            reordered.push_back(
                IfThenElse(loop->loop_var > loop->min, wait.stmt));
          } else {
            reordered.push_back(wait.stmt);
          }
        }
      }
      if (!removed.count(i)) {
        reordered.push_back(seq->seq[i]);
      }
    }
    return SeqStmt::Flatten(reordered);
  }

  std::vector<const ForNode *> loop_stack_;
};

class InitReusableBarriersRewriter : public StmtMutator {
public:
  Stmt operator()(Stmt body) {
    Stmt rewritten = VisitStmt(body);
    if (HasThreadExtent(rewritten)) {
      return rewritten;
    }
    return PrependBarrierInits(rewritten, {});
  }

private:
  class ThreadExtentFinder : public StmtVisitor {
  public:
    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == tir::attr::thread_extent) {
        found = true;
        return;
      }
      StmtVisitor::VisitStmt_(op);
    }

    bool found{false};
  };

  class BarrierMaskCollector : public StmtExprVisitor {
  public:
    explicit BarrierMaskCollector(std::vector<Var> scoped_vars)
        : scoped_vars_(std::move(scoped_vars)) {}

    void VisitStmt_(const ForNode *op) final {
      scoped_vars_.push_back(op->loop_var);
      StmtExprVisitor::VisitStmt_(op);
      scoped_vars_.pop_back();
    }

    void VisitStmt_(const LetStmtNode *op) final {
      scoped_vars_.push_back(op->var);
      StmtExprVisitor::VisitStmt_(op);
      scoped_vars_.pop_back();
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (const CallNode *call = op->value.as<CallNode>()) {
        if (call->op.same_as(barrier_arrive_and_wait()) &&
            !call->args.empty()) {
          BarrierMaskInfo info = BarrierMaskInfoFromArgs(call->args);
          ICHECK(!info.candidates.empty() ||
                 !ExprUsesAnyVar(info.expr, scoped_vars_))
              << "dynamic barrier mask depends on a local control-flow "
                 "variable and cannot be initialized in the enclosing entry "
                 "block required by suvm.barrier.init";
          AddUniqueBarrierMaskInfo(&masks, info);
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    std::vector<BarrierMaskInfo> masks;

  private:
    std::vector<Var> scoped_vars_;
  };

  static Stmt MakeBarrierInitStmt(const BarrierMaskInfo &participant_mask) {
    return Evaluate(Call(DataType::Handle(), barrier_init(),
                         MakeBarrierInitArgs(participant_mask)));
  }

  static bool HasThreadExtent(const Stmt &body) {
    ThreadExtentFinder finder;
    finder(body);
    return finder.found;
  }

  static Stmt PrependBarrierInits(const Stmt &body,
                                  const std::vector<Var> &scoped_vars) {
    BarrierMaskCollector collector(scoped_vars);
    collector(body);
    if (collector.masks.empty()) {
      return body;
    }

    Array<Stmt> stmts;
    for (const BarrierMaskInfo &mask : collector.masks) {
      stmts.push_back(MakeBarrierInitStmt(mask));
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const Stmt &stmt : seq->seq) {
        stmts.push_back(stmt);
      }
    } else {
      stmts.push_back(body);
    }
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tir::attr::thread_extent) {
      return StmtMutator::VisitStmt_(op);
    }

    Stmt body = StmtMutator::VisitStmt(op->body);
    if (HasThreadExtent(body)) {
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    return AttrStmt(op->node, op->attr_key, op->value,
                    PrependBarrierInits(body, scoped_vars_));
  }

  Stmt VisitStmt_(const ForNode *op) final {
    scoped_vars_.push_back(op->loop_var);
    Stmt body = StmtMutator::VisitStmt(op->body);
    scoped_vars_.pop_back();
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations, std::nullopt, op->span);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    scoped_vars_.push_back(op->var);
    Stmt body = StmtMutator::VisitStmt(op->body);
    scoped_vars_.pop_back();
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return LetStmt(op->var, op->value, body, op->span);
  }

  std::vector<Var> scoped_vars_;
};

// Main rewriter orchestrating the synchronization injection passes.
// It applies a sequence of passes: inject syncs, extract barriers, add device
// scope waits, and finally eliminate redundant synchronizations.
class SunmmioSyncRewriter : public IRMutatorWithAnalyzer {
public:
  SunmmioSyncRewriter(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  static PrimFunc Rewrite(PrimFunc f, arith::Analyzer *analyzer) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget).value();

    auto inject_sync_rewriter =
        InjectSyncRewriter(f->buffer_map, target, analyzer);
    f.CopyOnWrite()->body = inject_sync_rewriter(f->body);

    TokenBarrierMap token_to_barrier_mask =
        inject_sync_rewriter.get_token_to_barrier_mask();

    auto loop_missing_token_wait_rewriter = LoopMissingTokenWaitRewriter();
    f.CopyOnWrite()->body = loop_missing_token_wait_rewriter(f->body);

    // LoopMissingTokenWaitRewriter may add a wait immediately before reuse of
    // one logical token, after another command has already reused the same
    // physical engine.  Apply the engine-aware ordering only after all such
    // waits exist so none can consume newly launched work on that engine.
    auto move_loop_carried_engine_waits_rewriter =
        MoveLoopCarriedEngineWaitsRewriter();
    f.CopyOnWrite()->body = move_loop_carried_engine_waits_rewriter(f->body);

    auto device_func_wait_rewriter =
        DeviceFuncWaitRewriter(token_to_barrier_mask);
    f.CopyOnWrite()->body = device_func_wait_rewriter(f->body);

    auto eliminate_redundancy_rewriter = EliminateRedundancyRewriter(
        analyzer, std::vector<int>({}), token_to_barrier_mask);
    f.CopyOnWrite()->body = eliminate_redundancy_rewriter(f->body);

    auto ensure_all_gather_launch_barrier_rewriter =
        EnsureAllGatherLaunchBarrierRewriter(token_to_barrier_mask);
    f.CopyOnWrite()->body = ensure_all_gather_launch_barrier_rewriter(f->body);

    auto init_reusable_barriers_rewriter = InitReusableBarriersRewriter();
    f.CopyOnWrite()->body = init_reusable_barriers_rewriter(f->body);

    return f;
  }
};

// TVM transform pass entry point.
// Applies the SunmmioSyncRewriter to inject required synchronization
// primitives.
tvm::transform::Pass InjectSunmmioSync() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    if (!f->HasNonzeroAttr(tir::attr::kIsGlobalFunc)) {
      return f;
    }
    arith::Analyzer analyzer;
    return SunmmioSyncRewriter::Rewrite(f, &analyzer);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSunmmioSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectSunmmioSync", InjectSunmmioSync);
}

} // namespace tl
} // namespace tvm
