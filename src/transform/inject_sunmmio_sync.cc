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
 * \brief Inject hardware-unit synchronization and collective launch/completion
 * barriers.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
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
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/dist_comm.h"
#include "../op/utils.h"
#include "../target/sunmmio_access_range.h"
#include "../target/sunmmio_utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

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

bool SamePrimExpr(const PrimExpr &lhs, const PrimExpr &rhs) {
  return StructuralEqual()(lhs, rhs);
}

struct BarrierMaskInfo {
  PrimExpr expr;
  std::vector<int64_t> candidates;
};

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
      return div ? std::optional<int64_t>(a - *div * b)
                 : std::optional<int64_t>();
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
    return cond ? EvalInt64(*cond != 0 ? op->true_value : op->false_value,
                            analyzer)
                : std::optional<int64_t>();
  }
  if (const auto *call = expr.as<CallNode>()) {
    const auto *op = call->op.as<OpNode>();
    if (!op || call->args.size() != 2) {
      return std::nullopt;
    }
    if (op->name == "tir.bitwise_or") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) | UnsignedMask(b));
          });
    }
    if (op->name == "tir.bitwise_and") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) & UnsignedMask(b));
          });
    }
    if (op->name == "tir.bitwise_xor") {
      return eval_binary(
          call->args[0], call->args[1], [](int64_t a, int64_t b) {
            return static_cast<int64_t>(UnsignedMask(a) ^ UnsignedMask(b));
          });
    }
    if (op->name == "tir.shift_left") {
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
  return (mask & ~((uint64_t{1} << total_cores) - 1)) == 0;
}

bool MaskAlignedWithDirection(uint64_t mask, int direction, int mesh_nrow,
                              int mesh_ncol) {
  int total_cores = mesh_nrow * mesh_ncol;
  if (mask == 0 || !IsMaskWithinMesh(mask, total_cores)) {
    return false;
  }
  int min_participants =
      direction == 0 ? std::min(mesh_ncol, 2) : std::min(mesh_nrow, 2);
  if (CountMaskBits(mask) < min_participants) {
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
    } else if ((direction == 0 && row != ref_row) ||
               (direction == 1 && col != ref_col)) {
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

std::vector<int64_t> EnumerateMaskCandidates(PrimExpr expr, int direction,
                                             int mesh_nrow, int mesh_ncol,
                                             arith::Analyzer *analyzer) {
  VarCollector collector;
  collector(expr);
  if (collector.vars.empty()) {
    std::optional<int64_t> value = EvalInt64(expr, analyzer);
    if (value && MaskAlignedWithDirection(UnsignedMask(*value), direction,
                                          mesh_nrow, mesh_ncol)) {
      return {*value};
    }
    return {};
  }
  if (collector.vars.size() > 2) {
    return {};
  }

  int total_cores = mesh_nrow * mesh_ncol;
  int64_t num_cases = 1;
  for (size_t i = 0; i < collector.vars.size(); ++i) {
    num_cases *= total_cores;
  }

  std::vector<int64_t> candidates;
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
    if (MaskAlignedWithDirection(UnsignedMask(*value), direction, mesh_nrow,
                                 mesh_ncol)) {
      AddUniqueInt64(&candidates, *value);
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
  return SamePrimExpr(lhs.expr, rhs.expr) && lhs.candidates == rhs.candidates;
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
      << "broadcast_() call is missing its fixed argument prefix";
  size_t fixed_and_source_args = call->args.size();
  if (ParseSunmmioOdmaUnitExpr(call->args.back())) {
    --fixed_and_source_args;
  }
  ICHECK(fixed_and_source_args == static_cast<size_t>(kBroadcastArgCount) ||
         fixed_and_source_args == static_cast<size_t>(kBroadcastArgCount + 1))
      << "broadcast_() expects fixed args plus optional src_core";
  return fixed_and_source_args == static_cast<size_t>(kBroadcastArgCount + 1);
}

PrimExpr GetBroadcastSrcCore(const CallNode *call) {
  ICHECK(BroadcastCallHasSrcCore(call))
      << "broadcast_() call does not carry optional src_core";
  size_t index = call->args.size() - 1;
  if (ParseSunmmioOdmaUnitExpr(call->args.back())) {
    --index;
  }
  return call->args[index];
}

bool IsRsramBuffer(const Buffer &buffer) {
  return buffer.scope() == kSunmmioScopeRSRAM || buffer.scope() == "local";
}

class BufferLoadCollector : public ExprVisitor {
public:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsRsramBuffer(op->buffer)) {
      Region region;
      for (const PrimExpr &index : op->indices) {
        region.push_back(Range::FromMinExtent(index, 1));
      }
      loads.emplace_back(op->buffer, std::move(region));
    }
    ExprVisitor::VisitExpr_(op);
  }

  std::vector<std::pair<Buffer, Region>> loads;
};

enum class AccessProducerKind {
  kAsync,
  kTileLoad,
  kTileStore,
};

struct UnitAccess {
  Buffer buffer;
  Region region;
  bool write{false};
  AccessProducerKind producer_kind{AccessProducerKind::kAsync};
  SunmmioSyncUnits units{kSunmmioSyncNone};
};

// For small integer indices, constant translations stay disjoint modulo
// 2^bits even when neither endpoint has a useful absolute interval.
// Pending local.var reads are widened when their index storage is written.
static bool DisjointTranslatedSlots(const Range &lhs, const Range &rhs) {
  auto dtype = lhs->min.dtype();
  auto a = FiniteWidthInteger::Literal(lhs->extent),
       b = FiniteWidthInteger::Literal(rhs->extent);
  if (dtype != rhs->min.dtype() || !dtype.is_scalar() || dtype.bits() > 32 ||
      !a || !b || *a <= 0 || *b <= 0)
    return false;
  uint64_t period = uint64_t{1} << dtype.bits();
  struct Slot {
    PrimExpr base;
    uint64_t scale, offset;
  };
  auto slot = [&](PrimExpr expr) {
    uint64_t scale = 1, offset = 0;
    while (expr.dtype() == dtype) {
      if (auto *add = expr.as<AddNode>()) {
        if (auto n = FiniteWidthInteger::Literal(add->b)) {
          offset += scale * static_cast<uint64_t>(*n);
          expr = add->a;
          continue;
        }
        if (auto n = FiniteWidthInteger::Literal(add->a)) {
          offset += scale * static_cast<uint64_t>(*n);
          expr = add->b;
          continue;
        }
      }
      if (auto *sub = expr.as<SubNode>())
        if (auto n = FiniteWidthInteger::Literal(sub->b)) {
          offset -= scale * static_cast<uint64_t>(*n);
          expr = sub->a;
          continue;
        }
      if (auto *mul = expr.as<MulNode>())
        if (auto n = FiniteWidthInteger::Literal(mul->b)) {
          scale *= static_cast<uint64_t>(*n);
          expr = mul->a;
          continue;
        }
      break;
    }
    return Slot{expr, scale % period, offset % period};
  };
  auto x = slot(lhs->min), y = slot(rhs->min);
  const auto *load = x.base.as<BufferLoadNode>();
  if (!(x.base.as<VarNode>() ||
        (load && load->buffer.scope() == "local.var" &&
         std::all_of(
             load->indices.begin(), load->indices.end(),
             [](const PrimExpr &e) { return e.as<IntImmNode>() != nullptr; }))))
    return false;
  if (x.scale != y.scale || !StructuralEqual()(x.base, y.base))
    return false;
  uint64_t delta = (y.offset - x.offset) % period;
  return delta >= *a && period - delta >= *b;
}

bool AccessRegionsMayOverlap(const UnitAccess &lhs, const UnitAccess &rhs) {
  if (!lhs.buffer->data.same_as(rhs.buffer->data))
    return false;
  if (!lhs.buffer.same_as(rhs.buffer) &&
      !StructuralEqual()(lhs.buffer, rhs.buffer))
    return true;
  if (lhs.region.size() != rhs.region.size())
    return true;
  arith::Analyzer analyzer;
  FiniteWidthInteger facts(&analyzer);
  for (size_t i = 0; i < lhs.region.size(); ++i) {
    if (DisjointTranslatedSlots(lhs.region[i], rhs.region[i]))
      return false;
    auto a =
        AnalyzeSunmmioAccessRange(&facts, lhs.region[i]->min,
                                  lhs.region[i]->extent, lhs.buffer->shape[i]);
    auto b =
        AnalyzeSunmmioAccessRange(&facts, rhs.region[i]->min,
                                  rhs.region[i]->extent, rhs.buffer->shape[i]);
    if (!dist_transform::RegionsMayOverlap({a.envelope}, {b.envelope}) ||
        !dist_transform::RegionsMayOverlap({a.current}, {b.current}))
      return false;
  }
  return true;
}

bool NeedsUnitSync(const UnitAccess &producer, const UnitAccess &consumer) {
  if (!producer.write && !consumer.write) {
    return false;
  }
  if (producer.producer_kind == AccessProducerKind::kAsync) {
    return true;
  }
  if (consumer.producer_kind != AccessProducerKind::kAsync) {
    return false;
  }
  if (producer.producer_kind == AccessProducerKind::kTileStore) {
    return true;
  }
  return producer.producer_kind == AccessProducerKind::kTileLoad &&
         consumer.write;
}

class InjectUnitSyncRewriter : public StmtMutator {
public:
  explicit InjectUnitSyncRewriter(const PrimFunc &func)
      : layouts_(func->GetAttr<Map<Buffer, Layout>>(attr::kLayoutMap)
                     .value_or(Map<Buffer, Layout>())),
        config_(GetSunmmioTileProcessorConfig(
            func->GetAttr<Target>(tvm::attr::kTarget))) {
    class Footprints : public StmtVisitor {
    public:
      explicit Footprints(InjectUnitSyncRewriter *owner) : owner_(owner) {}

    private:
      void VisitStmt_(const ForNode *op) final {
        ctx_.Enter(op);
        VisitStmt(op->body);
        ctx_.Leave(op);
      }
      void VisitStmt_(const LetStmtNode *op) final {
        if (SideEffect(op->value) <= CallEffectKind::kPure)
          ctx_.definitions[op->var.get()] = op->value;
        VisitStmt(op->body);
        ctx_.definitions.erase(op->var.get());
      }
      void VisitStmt_(const BufferStoreNode *op) final {
        owner_->store_regions_.emplace(
            op, GetSunmmioScalarTileAccessRegion(
                    op->buffer, op->indices, owner_->layouts_, owner_->config_,
                    &ctx_.analyzer, &ctx_.definitions,
                    ctx_.LogicalShape(op->indices)));
      }
      SunmmioAccessContext ctx_;
      InjectUnitSyncRewriter *owner_;
    } footprints(this);
    footprints(func->body);
  }
  Stmt operator()(Stmt body) {
    Stmt rewritten = VisitStmt(body);
    SunmmioSyncUnits pending_links = PendingUnits() & kLinkUnits;
    CompleteUnits(pending_links);
    return AppendSync(std::move(rewritten), pending_links);
  }

private:
  using Accesses = std::vector<UnitAccess>;

  struct FlowState {
    Accesses pending;
    bool reachable{true};
  };

  struct LoopFrame {
    FlowState breaks{{}, false};
    std::vector<std::pair<const EvaluateNode *, FlowState>> continues;
    size_t allocation_depth;
  };

  struct LoopInfo {
    Var var{ffi::ObjectPtr<VarNode>()};
    Range domain;
    PrimExpr step;
    std::unordered_set<const VarNode *> local_vars;
    Map<Var, PrimExpr> let_values;
  };

  static void MergeAccesses(Accesses *dst, const Accesses &src) {
    for (const UnitAccess &access : src) {
      bool found = false;
      for (const UnitAccess &existing : *dst) {
        found |= existing.buffer.same_as(access.buffer) &&
                 StructuralEqual()(existing.region, access.region) &&
                 existing.write == access.write &&
                 existing.producer_kind == access.producer_kind &&
                 existing.units == access.units;
      }
      if (!found) {
        dst->push_back(access);
      }
    }
  }

  static FlowState Join(FlowState lhs, const FlowState &rhs) {
    if (!lhs.reachable) {
      return rhs;
    }
    if (rhs.reachable) {
      MergeAccesses(&lhs.pending, rhs.pending);
    }
    return lhs;
  }

  FlowState GetState() const { return {pending_, reachable_}; }

  void SetState(const FlowState &state) {
    pending_ = state.pending;
    reachable_ = state.reachable;
  }

  static PrimExpr ResolveLoopLets(const PrimExpr &expr, const LoopInfo &info) {
    class Resolver : public ExprMutator {
    public:
      explicit Resolver(const Map<Var, PrimExpr> &values) : values_(values) {}

    private:
      PrimExpr VisitExpr_(const VarNode *op) final {
        Var var = ffi::GetRef<Var>(op);
        auto value = values_.Get(var);
        return value ? VisitExpr(value.value()) : var;
      }
      const Map<Var, PrimExpr> &values_;
    } resolver(info.let_values);
    return resolver(expr);
  }

  static bool HasUnknownLoopIndex(const PrimExpr &expr, const LoopInfo &info) {
    bool unknown = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        unknown |= info.local_vars.count(var) && var != info.var.get();
      }
      unknown |= node.as<BufferLoadNode>() != nullptr;
      if (const auto *call = node.as<CallNode>()) {
        unknown |=
            SideEffect(ffi::GetRef<PrimExpr>(call)) > CallEffectKind::kPure;
      }
    });
    return unknown;
  }

  static LoopInfo MakeLoopInfo(const Stmt &body, const ForNode *loop) {
    LoopInfo info;
    if (loop) {
      info.var = loop->loop_var;
      info.domain = Range::FromMinExtent(loop->min, loop->extent);
      info.step = loop->step.value_or(make_const(loop->loop_var.dtype(), 1));
      info.local_vars.insert(loop->loop_var.get());
    }
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *nested = node.as<ForNode>()) {
        info.local_vars.insert(nested->loop_var.get());
      } else if (const auto *let = node.as<LetStmtNode>()) {
        info.local_vars.insert(let->var.get());
        if (!dist_transform::RouteExprBufferLoadDetector().Detect(let->value) &&
            tir::SideEffect(let->value) <= tir::CallEffectKind::kPure) {
          info.let_values.Set(let->var, let->value);
        }
      } else if (const auto *let = node.as<LetNode>()) {
        info.local_vars.insert(let->var.get());
        if (!dist_transform::RouteExprBufferLoadDetector().Detect(let->value) &&
            SideEffect(let->value) <= CallEffectKind::kPure) {
          info.let_values.Set(let->var, let->value);
        }
      }
    });
    return info;
  }

  static Optional<Range> SafeLoopDomain(const LoopInfo &info) {
    if (!info.var.defined() || HasUnknownLoopIndex(info.domain->min, info) ||
        HasUnknownLoopIndex(info.domain->extent, info))
      return std::nullopt;
    arith::Analyzer analyzer;
    FiniteWidthInteger facts(&analyzer);
    auto start = facts.Analyze(info.domain->min),
         extent = facts.Analyze(info.domain->extent);
    auto type = facts.Analyze(info.var);
    auto hi = start.range.hi + extent.range.hi - 1;
    auto span = hi - start.range.lo + 1;
    if (!start.valid || !extent.valid || extent.range.hi <= 0 ||
        start.range.lo < type.range.lo || hi > type.range.hi ||
        start.range.lo <= INT64_MIN || hi >= INT64_MAX || span > INT64_MAX)
      return std::nullopt;
    return Range::FromMinExtent(I64Imm(static_cast<int64_t>(start.range.lo)),
                                I64Imm(static_cast<int64_t>(span)));
  }

  static UnitAccess CoverLoopAccess(UnitAccess access, const LoopInfo &info) {
    Region region;
    std::unordered_map<const VarNode *, arith::IntSet> domains;
    auto domain = SafeLoopDomain(info);
    arith::Analyzer analyzer;
    if (info.var.defined()) {
      domains.emplace(info.var.get(),
                      domain ? arith::IntSet::FromRange(domain.value())
                             : arith::IntSet::Everything());
      if (domain)
        analyzer.Bind(info.var, domain.value());
    }
    FiniteWidthInteger facts(&analyzer);
    for (size_t dim = 0; dim < access.region.size(); ++dim) {
      PrimExpr min = ResolveLoopLets(access.region[dim]->min, info);
      PrimExpr extent = ResolveLoopLets(access.region[dim]->extent, info);
      Range full =
          Range::FromMinExtent(make_zero(access.buffer->shape[dim].dtype()),
                               access.buffer->shape[dim]);
      region.push_back(
          HasUnknownLoopIndex(min, info) || HasUnknownLoopIndex(extent, info)
              ? full
              : arith::EvalSet(
                    AnalyzeSunmmioAccessRange(&facts, min, extent,
                                              access.buffer->shape[dim])
                        .current,
                    domains)
                    .CoverRange(full));
    }
    access.region = std::move(region);
    return access;
  }

  static FlowState CoverLoopState(FlowState state, const LoopInfo &info) {
    Accesses covered;
    for (const UnitAccess &access : state.pending) {
      MergeAccesses(&covered, {CoverLoopAccess(access, info)});
    }
    state.pending = std::move(covered);
    return state;
  }

  static bool OverlapAcrossIterations(const UnitAccess &producer,
                                      const UnitAccess &consumer,
                                      const LoopInfo &info) {
    if (!producer.buffer->data.same_as(consumer.buffer->data)) {
      return false;
    }
    if (!producer.buffer.same_as(consumer.buffer) &&
        !StructuralEqual()(producer.buffer, consumer.buffer)) {
      return true;
    }
    if (producer.region.size() != consumer.region.size()) {
      return true;
    }
    arith::Analyzer analyzer;
    Map<Var, PrimExpr> later_iteration;
    if (info.var.defined()) {
      const auto *step = info.step.as<IntImmNode>();
      auto domain = SafeLoopDomain(info);
      if (domain && step && step->value > 0) {
        Var distance("sync_iteration_distance", info.var.dtype());
        analyzer.Bind(info.var, domain.value());
        analyzer.Bind(distance,
                      Range::FromMinExtent(make_const(distance.dtype(), 1),
                                           domain.value()->extent));
        later_iteration.Set(info.var, info.var + info.step * distance);
      }
    }
    for (size_t dim = 0; dim < producer.region.size(); ++dim) {
      PrimExpr pmin = ResolveLoopLets(producer.region[dim]->min, info);
      PrimExpr pextent = ResolveLoopLets(producer.region[dim]->extent, info);
      PrimExpr cmin = ResolveLoopLets(consumer.region[dim]->min, info);
      PrimExpr cextent = ResolveLoopLets(consumer.region[dim]->extent, info);
      if (HasUnknownLoopIndex(pmin, info) ||
          HasUnknownLoopIndex(pextent, info) ||
          HasUnknownLoopIndex(cmin, info) ||
          HasUnknownLoopIndex(cextent, info)) {
        continue;
      }
      if (info.var.defined() && later_iteration.empty()) {
        continue;
      }
      cmin = Substitute(cmin, later_iteration);
      cextent = Substitute(cextent, later_iteration);
      FiniteWidthInteger facts(&analyzer);
      auto p = AnalyzeSunmmioAccessRange(&facts, pmin, pextent,
                                         producer.buffer->shape[dim])
                   .current;
      auto c = AnalyzeSunmmioAccessRange(&facts, cmin, cextent,
                                         consumer.buffer->shape[dim])
                   .current;
      if (analyzer.CanProve(p->min + p->extent <= c->min) ||
          analyzer.CanProve(c->min + c->extent <= p->min)) {
        return false;
      }
    }
    return true;
  }

  static constexpr SunmmioSyncUnits kTileMemoryUnits =
      kSunmmioSyncVector | kSunmmioSyncRsram;
  static constexpr SunmmioSyncUnits kLinkUnits =
      kSunmmioSyncHlink | kSunmmioSyncVlink;

  static Stmt MakeSync(SunmmioSyncUnits units) {
    ICHECK_NE(units, kSunmmioSyncNone);
    return Evaluate(Call(DataType::Handle(), sunmmio_sync(),
                         {IntImm(DataType::Int(32), units)}));
  }

  static Stmt PrependSync(Stmt stmt, SunmmioSyncUnits units) {
    if (units == kSunmmioSyncNone) {
      return stmt;
    }
    return SeqStmt::Flatten(Array<Stmt>{MakeSync(units), std::move(stmt)});
  }

  static Stmt AppendSync(Stmt stmt, SunmmioSyncUnits units) {
    if (units == kSunmmioSyncNone) {
      return stmt;
    }
    return SeqStmt::Flatten(Array<Stmt>{std::move(stmt), MakeSync(units)});
  }

  static UnitAccess MakeRegionAccess(const BufferRegion &region, bool write,
                                     AccessProducerKind kind,
                                     SunmmioSyncUnits units) {
    return {region->buffer, region->region, write, kind, units};
  }

  static SunmmioSyncUnits OdmaUnits(const CallNode *call) {
    std::optional<SunmmioOdmaUnit> unit = GetSunmmioOdmaUnit(call);
    if (!unit) {
      // ResolveSunmmioUnit runs before this pass in the production pipeline.
      // Standalone pass use remains conservative when the marker is absent.
      return kSunmmioSyncOdma0 | kSunmmioSyncOdma1;
    }
    return *unit == SunmmioOdmaUnit::kOdma0 ? kSunmmioSyncOdma0
                                            : kSunmmioSyncOdma1;
  }

  static bool GetAsyncAccesses(const CallNode *call,
                               std::vector<UnitAccess> *accesses,
                               std::vector<UnitAccess> *argument_loads,
                               bool *completed_inline) {
    *completed_inline = false;
    if (call->op.same_as(dma_copy()) ||
        call->op.same_as(sunmmio_layout_transform()) ||
        call->op.same_as(sunmmio_transpose())) {
      ICHECK_GE(call->args.size(), 2U);
      CollectRegionArgumentLoads(call, {0, 1}, argument_loads);
      SunmmioSyncUnits units = OdmaUnits(call);
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[0]), false,
                           AccessProducerKind::kAsync, units));
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[1]), true,
                           AccessProducerKind::kAsync, units));
      return true;
    }
    if (call->op.same_as(mma_sunmmio())) {
      ICHECK_GE(call->args.size(), 3U);
      CollectRegionArgumentLoads(call, {0, 1, 2}, argument_loads);
      for (int index : {0, 1, 2}) {
        accesses->push_back(
            MakeRegionAccess(NormalizeToBufferRegion(call->args[index]), false,
                             AccessProducerKind::kAsync, kSunmmioSyncTc));
      }
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[2]), true,
                           AccessProducerKind::kAsync, kSunmmioSyncTc));
      return true;
    }
    if (call->op.same_as(broadcast_())) {
      ICHECK_GE(call->args.size(), static_cast<size_t>(kBroadcastArgCount));
      CollectRegionArgumentLoads(call,
                                 {static_cast<size_t>(kBroadcastArgSrc),
                                  static_cast<size_t>(kBroadcastArgDst)},
                                 argument_loads);
      const auto *direction =
          call->args[kBroadcastArgDirection].as<IntImmNode>();
      ICHECK(direction && (direction->value == 0 || direction->value == 1))
          << "tl.broadcast_ requires a constant row/column direction";
      SunmmioSyncUnits units =
          direction->value == 0 ? kSunmmioSyncHlink : kSunmmioSyncVlink;
      accesses->push_back(MakeRegionAccess(
          NormalizeToBufferRegion(call->args[kBroadcastArgSrc]), false,
          AccessProducerKind::kAsync, units));
      accesses->push_back(MakeRegionAccess(
          NormalizeToBufferRegion(call->args[kBroadcastArgDst]), true,
          AccessProducerKind::kAsync, units));
      // SUVM codegen keeps multicast and its link sync in the same source-core
      // branch. Unguarded multicasts remain pending until the post-collective
      // barrier, which completes the link before all participant cores leave.
      *completed_inline = BroadcastCallHasSrcCore(call);
      return true;
    }
    return false;
  }

  static bool GetTileCompoundAccesses(const CallNode *call,
                                      std::vector<UnitAccess> *accesses,
                                      std::vector<UnitAccess> *argument_loads) {
    bool is_pack = call->op.same_as(mx_pack());
    bool is_unpack = call->op.same_as(mx_unpack());
    if (!is_pack && !is_unpack) {
      return false;
    }

    ICHECK_EQ(call->args.size(), 3U);
    CollectRegionArgumentLoads(call, {0, 1, 2}, argument_loads);
    // MX lowering expands to tile loads followed by tile stores. Preserve those
    // access kinds so a later tokenless async operation drains vector/RSRAM.
    for (size_t i = 0; i < call->args.size(); ++i) {
      bool write = is_pack ? i == 2 : i != 0;
      accesses->push_back(
          MakeRegionAccess(NormalizeToBufferRegion(call->args[i]), write,
                           write ? AccessProducerKind::kTileStore
                                 : AccessProducerKind::kTileLoad,
                           kTileMemoryUnits));
    }
    return true;
  }

  static void CollectWaitAnyLoads(const CallNode *call,
                                  std::vector<UnitAccess> *accesses,
                                  const Map<Buffer, Layout> &layouts,
                                  const SunmmioTileProcessorConfig &config) {
    CollectRegionArgumentLoads(call, {3, 4}, accesses);
    for (size_t position : {size_t{3}, size_t{4}}) {
      BufferRegion region = NormalizeToBufferRegion(call->args[position]);
      ICHECK_EQ(region->region.size(), 1U);
      const Range &logical = region->region[0];
      Region first = GetSunmmioScalarTileAccessRegion(
          region->buffer, {logical->min}, layouts, config);
      Region last = GetSunmmioScalarTileAccessRegion(
          region->buffer, {logical->min + logical->extent - 1}, layouts,
          config);
      arith::Analyzer analyzer;
      Region physical{Range::FromMinExtent(
          first[0]->min,
          analyzer.Simplify(last[0]->min + last[0]->extent - first[0]->min))};
      accesses->push_back({region->buffer, std::move(physical), false,
                           AccessProducerKind::kTileLoad, kTileMemoryUnits});
    }
  }

  static void CollectTileLoads(const PrimExpr &expr,
                               std::vector<UnitAccess> *accesses) {
    BufferLoadCollector collector;
    collector(expr);
    for (const auto &[buffer, region] : collector.loads) {
      accesses->push_back({buffer, region, false, AccessProducerKind::kTileLoad,
                           kTileMemoryUnits});
    }
  }

  static void
  CollectRegionArgumentLoads(const CallNode *call,
                             std::initializer_list<size_t> region_args,
                             std::vector<UnitAccess> *accesses) {
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (std::find(region_args.begin(), region_args.end(), i) ==
          region_args.end()) {
        CollectTileLoads(call->args[i], accesses);
        continue;
      }
      BufferRegion region = NormalizeToBufferRegion(call->args[i]);
      for (const Range &range : region->region) {
        CollectTileLoads(range->min, accesses);
        CollectTileLoads(range->extent, accesses);
      }
    }
  }

  static void CollectRangeLoads(const Region &region,
                                std::vector<UnitAccess> *accesses) {
    for (const Range &range : region) {
      CollectTileLoads(range->min, accesses);
      CollectTileLoads(range->extent, accesses);
    }
  }

  Accesses CollectLoopAccesses(const Stmt &body) const {
    class Collector : public StmtExprVisitor {
    public:
      Collector(
          const Map<Buffer, Layout> &layouts,
          const SunmmioTileProcessorConfig &config,
          const std::unordered_map<const BufferStoreNode *, Region> &regions)
          : layouts_(layouts), config_(config), regions_(regions) {}
      Accesses accesses;

    private:
      void VisitStmt_(const SeqStmtNode *op) final {
        for (const Stmt &stmt : op->seq) {
          VisitStmt(stmt);
          const auto *evaluate = stmt.as<EvaluateNode>();
          const auto *call =
              evaluate ? evaluate->value.as<CallNode>() : nullptr;
          if (call && (call->op.same_as(builtin::break_loop()) ||
                       call->op.same_as(builtin::continue_loop()) ||
                       call->op.same_as(builtin::ret()))) {
            break;
          }
        }
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        if (IsRsramBuffer(op->buffer)) {
          Region region = regions_.at(op);
          accesses.push_back({op->buffer, std::move(region), true,
                              AccessProducerKind::kTileStore,
                              kTileMemoryUnits});
        }
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const BufferLoadNode *op) final {
        if (IsRsramBuffer(op->buffer)) {
          Region region;
          for (const PrimExpr &index : op->indices) {
            region.push_back(Range::FromMinExtent(index, 1));
          }
          accesses.push_back({op->buffer, std::move(region), false,
                              AccessProducerKind::kTileLoad, kTileMemoryUnits});
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(dist_wait_any_())) {
          CollectWaitAnyLoads(op, &accesses, layouts_, config_);
          return;
        }
        Accesses args;
        bool completed_inline = false;
        if (GetAsyncAccesses(op, &accesses, &args, &completed_inline) ||
            GetTileCompoundAccesses(op, &accesses, &args)) {
          MergeAccesses(&accesses, args);
          return;
        }
        if (op->op.same_as(dist_put_())) {
          CollectRegionArgumentLoads(op, {0, 1}, &accesses);
          accesses.push_back(
              MakeRegionAccess(NormalizeToBufferRegion(op->args[0]), false,
                               AccessProducerKind::kAsync, kSunmmioSyncNone));
          return;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      const Map<Buffer, Layout> &layouts_;
      const SunmmioTileProcessorConfig &config_;
      const std::unordered_map<const BufferStoreNode *, Region> &regions_;
    } collector(layouts_, config_, store_regions_);
    collector(body);
    return std::move(collector.accesses);
  }

  SunmmioSyncUnits CompleteAllocationAccesses(size_t first) {
    SunmmioSyncUnits units = kSunmmioSyncNone;
    for (const UnitAccess &access : pending_) {
      for (size_t index = first; index < active_allocations_.size(); ++index) {
        if (access.buffer->data.same_as(active_allocations_[index])) {
          units |= access.units;
        }
      }
    }
    CompleteUnits(units);
    return units;
  }

  static void CompleteStateUnits(FlowState *state, SunmmioSyncUnits units) {
    auto &pending = state->pending;
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](const UnitAccess &access) {
                                   return (units & access.units) ==
                                          access.units;
                                 }),
                  pending.end());
  }

  static SunmmioSyncUnits CarriedSync(const FlowState &state,
                                      const Accesses &consumers,
                                      const LoopInfo &info) {
    SunmmioSyncUnits units = kSunmmioSyncNone;
    if (info.var.defined()) {
      arith::Analyzer analyzer;
      FiniteWidthInteger facts(&analyzer);
      auto step = facts.Analyze(info.step),
           extent = facts.Analyze(info.domain->extent);
      if (step.valid && extent.valid && step.range.lo > 0 &&
          extent.range.hi <= step.range.lo) {
        return units;
      }
    }
    if (state.reachable) {
      for (const UnitAccess &producer : state.pending) {
        for (const UnitAccess &consumer : consumers) {
          if (NeedsUnitSync(producer, consumer) &&
              OverlapAcrossIterations(producer, consumer, info)) {
            units |= producer.units;
          }
        }
      }
    }
    return units;
  }

  static Stmt AddContinueSyncs(
      Stmt body,
      const std::unordered_map<const EvaluateNode *, SunmmioSyncUnits> &waits) {
    class Rewriter : public StmtMutator {
    public:
      explicit Rewriter(const std::unordered_map<const EvaluateNode *,
                                                 SunmmioSyncUnits> &waits)
          : waits_(waits) {}

    private:
      Stmt VisitStmt_(const EvaluateNode *op) final {
        auto found = waits_.find(op);
        return found == waits_.end()
                   ? ffi::GetRef<Stmt>(op)
                   : PrependSync(ffi::GetRef<Stmt>(op), found->second);
      }
      const std::unordered_map<const EvaluateNode *, SunmmioSyncUnits> &waits_;
    } rewriter(waits);
    return rewriter(std::move(body));
  }

  SunmmioSyncUnits RequiredSync(const std::vector<UnitAccess> &accesses) const {
    SunmmioSyncUnits required = kSunmmioSyncNone;
    for (const UnitAccess &consumer : accesses) {
      for (const UnitAccess &producer : pending_) {
        if (!NeedsUnitSync(producer, consumer) ||
            !AccessRegionsMayOverlap(producer, consumer)) {
          continue;
        }
        required |= producer.units;
      }
    }
    return required;
  }

  void CompleteUnits(SunmmioSyncUnits completed) {
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                  [&](const UnitAccess &access) {
                                    return (completed & access.units) ==
                                           access.units;
                                  }),
                   pending_.end());
  }

  void RecordAccesses(const std::vector<UnitAccess> &accesses) {
    pending_.insert(pending_.end(), accesses.begin(), accesses.end());
  }

  void InvalidatePendingIndices(const std::vector<BufferRegion> &writes) {
    for (UnitAccess &access : pending_) {
      access.region = dist_transform::WidenRegionForIndexWrites(
          access.buffer, access.region, writes,
          [](const Buffer &a, const Region &ar, const Buffer &b,
             const Region &br) {
            return AccessRegionsMayOverlap({a, ar}, {b, br});
          });
    }
  }

  void InvalidatePendingIndices(const Accesses &accesses) {
    std::vector<BufferRegion> writes;
    for (const UnitAccess &access : accesses) {
      if (access.write) {
        writes.emplace_back(access.buffer, access.region);
      }
    }
    InvalidatePendingIndices(writes);
  }

  SunmmioSyncUnits ProcessAccesses(const std::vector<UnitAccess> &accesses,
                                   bool completed_inline = false) {
    SunmmioSyncUnits required = RequiredSync(accesses);
    CompleteUnits(required);
    if (!completed_inline) {
      RecordAccesses(accesses);
    }
    return required;
  }

  SunmmioSyncUnits
  SynchronizeBeforeLoads(const std::vector<UnitAccess> &accesses) {
    SunmmioSyncUnits required = RequiredSync(accesses);
    CompleteUnits(required);
    return required;
  }

  SunmmioSyncUnits DrainPending() {
    SunmmioSyncUnits units = PendingUnits();
    CompleteUnits(units);
    return units;
  }

  SunmmioSyncUnits PendingUnits() const {
    SunmmioSyncUnits units = kSunmmioSyncNone;
    for (const UnitAccess &access : pending_) {
      units |= access.units;
    }
    return units;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> sequence;
    for (const Stmt &stmt : op->seq) {
      Stmt rewritten = reachable_ ? VisitStmt(stmt) : stmt;
      if (const auto *nested = rewritten.as<SeqStmtNode>()) {
        for (const Stmt &child : nested->seq) {
          sequence.push_back(child);
        }
      } else {
        sequence.push_back(rewritten);
      }
    }
    return SeqStmt::Flatten(sequence);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(builtin::ret())) {
      SunmmioSyncUnits units = DrainPending();
      reachable_ = false;
      return PrependSync(ffi::GetRef<Stmt>(op), units);
    }
    if (call && (call->op.same_as(builtin::break_loop()) ||
                 call->op.same_as(builtin::continue_loop()))) {
      ICHECK(!loop_stack_.empty()) << "Unit sync: loop jump outside a loop";
      SunmmioSyncUnits units =
          CompleteAllocationAccesses(loop_stack_.back().allocation_depth);
      if (call->op.same_as(builtin::break_loop())) {
        loop_stack_.back().breaks =
            Join(std::move(loop_stack_.back().breaks), GetState());
      } else {
        loop_stack_.back().continues.emplace_back(op, GetState());
      }
      reachable_ = false;
      return PrependSync(ffi::GetRef<Stmt>(op), units);
    }
    if (call && call->op.same_as(sunmmio_sync())) {
      ICHECK_EQ(call->args.size(), 1U);
      const auto *units = call->args[0].as<IntImmNode>();
      ICHECK(units) << "tl.sunmmio_sync expects a constant unit mask";
      CompleteUnits(static_cast<SunmmioSyncUnits>(units->value));
      return StmtMutator::VisitStmt_(op);
    }

    if (call && call->op.same_as(barrier_arrive_and_wait())) {
      SunmmioSyncUnits completed_links = PendingUnits() & kLinkUnits;
      CompleteUnits(completed_links);
      return PrependSync(StmtMutator::VisitStmt_(op), completed_links);
    }

    if (call && call->op.same_as(dist_wait_any_())) {
      Accesses reads;
      CollectWaitAnyLoads(call, &reads, layouts_, config_);
      SunmmioSyncUnits required = ProcessAccesses(reads);
      return PrependSync(StmtMutator::VisitStmt_(op), required);
    }
    if (call && call->op.same_as(dist_wait_signal_())) {
      ICHECK_EQ(call->args.size(), 3U);
      // A signal wait acquires remote writes without naming a destination
      // region. Complete all pending local accesses before the acquire.
      SunmmioSyncUnits required = DrainPending();
      return PrependSync(StmtMutator::VisitStmt_(op), required);
    }

    if (call && call->op.same_as(dist_put_())) {
      ICHECK_EQ(call->args.size(), 6U);
      std::vector<UnitAccess> argument_loads;
      CollectRegionArgumentLoads(call, {0, 1}, &argument_loads);
      SunmmioSyncUnits required = SynchronizeBeforeLoads(argument_loads);
      std::vector<UnitAccess> source_accesses{
          MakeRegionAccess(NormalizeToBufferRegion(call->args[0]), false,
                           AccessProducerKind::kAsync, kSunmmioSyncNone)};
      required |= SynchronizeBeforeLoads(source_accesses);
      RecordAccesses(argument_loads);
      return PrependSync(StmtMutator::VisitStmt_(op), required);
    }

    // Dist leaf operations own a separate completion protocol. Until that
    // protocol is modeled in unit synchronization, leave their encoded buffer
    // regions untouched instead of treating their vector arguments as loads.
    if (call && (call->op.same_as(dist_wait_send()) ||
                 call->op.same_as(dist_expect_()))) {
      return StmtMutator::VisitStmt_(op);
    }

    std::vector<UnitAccess> accesses;
    std::vector<UnitAccess> argument_loads;
    bool completed_inline = false;
    if (call &&
        GetAsyncAccesses(call, &accesses, &argument_loads, &completed_inline)) {
      SunmmioSyncUnits required = SynchronizeBeforeLoads(argument_loads);
      required |= ProcessAccesses(accesses, completed_inline);
      // Argument expressions are evaluated before the async submission. Keep
      // their memory accesses pending for later statements without introducing
      // a same-statement dependency that could only be synchronized too early.
      RecordAccesses(argument_loads);
      InvalidatePendingIndices(accesses);
      return PrependSync(StmtMutator::VisitStmt_(op), required);
    }

    if (call && GetTileCompoundAccesses(call, &accesses, &argument_loads)) {
      SunmmioSyncUnits required = SynchronizeBeforeLoads(argument_loads);
      required |= ProcessAccesses(accesses);
      RecordAccesses(argument_loads);
      InvalidatePendingIndices(accesses);
      return PrependSync(StmtMutator::VisitStmt_(op), required);
    }

    CollectTileLoads(op->value, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    std::vector<UnitAccess> accesses;
    for (const PrimExpr &index : op->indices) {
      CollectTileLoads(index, &accesses);
    }
    CollectTileLoads(op->value, &accesses);
    if (op->predicate.defined()) {
      CollectTileLoads(op->predicate.value(), &accesses);
    }
    Region region = store_regions_.at(op);
    if (IsRsramBuffer(op->buffer)) {
      accesses.push_back({op->buffer, region, true,
                          AccessProducerKind::kTileStore, kTileMemoryUnits});
    }
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    // local.var stores do not issue unit work, but can change the addresses of
    // earlier pending operations just like writes to an index tensor can.
    InvalidatePendingIndices(
        std::vector<BufferRegion>{BufferRegion(op->buffer, std::move(region))});
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    std::vector<UnitAccess> accesses;
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(dist_wait_any_())) {
      CollectWaitAnyLoads(call, &accesses, layouts_, config_);
    } else {
      CollectTileLoads(op->value, &accesses);
    }
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    Stmt body = VisitStmt(op->body);
    LoopInfo scope;
    scope.local_vars.insert(op->var.get());
    if (!dist_transform::RouteExprBufferLoadDetector().Detect(op->value) &&
        SideEffect(op->value) <= CallEffectKind::kPure) {
      scope.let_values.Set(op->var, op->value);
    }
    SetState(CoverLoopState(GetState(), scope));
    return PrependSync(LetStmt(op->var, op->value, body, op->span), required);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    std::vector<UnitAccess> accesses;
    CollectTileLoads(op->value, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const AssertStmtNode *op) final {
    std::vector<UnitAccess> accesses;
    CollectTileLoads(op->condition, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    std::vector<UnitAccess> accesses;
    for (const PrimExpr &extent : op->extents) {
      CollectTileLoads(extent, &accesses);
    }
    CollectTileLoads(op->condition, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    active_allocations_.push_back(op->buffer_var);
    Stmt body = VisitStmt(op->body);
    if (reachable_) {
      body = AppendSync(std::move(body), CompleteAllocationAccesses(
                                             active_allocations_.size() - 1));
    }
    active_allocations_.pop_back();
    return PrependSync(Allocate(op->buffer_var, op->dtype, op->extents,
                                op->condition, body, op->annotations, op->span),
                       required);
  }

  Stmt VisitStmt_(const BufferRealizeNode *op) final {
    std::vector<UnitAccess> accesses;
    CollectRangeLoads(op->bounds, &accesses);
    CollectTileLoads(op->condition, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    std::vector<UnitAccess> accesses;
    for (const PrimExpr &value : op->iter_values) {
      CollectTileLoads(value, &accesses);
    }
    CollectTileLoads(op->predicate, &accesses);
    SunmmioSyncUnits required = ProcessAccesses(accesses);
    return PrependSync(StmtMutator::VisitStmt_(op), required);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    std::vector<UnitAccess> condition_loads;
    CollectTileLoads(op->condition, &condition_loads);
    SunmmioSyncUnits prefix = ProcessAccesses(condition_loads);
    FlowState entry = GetState();
    Stmt then_case = VisitStmt(op->then_case);
    FlowState then_state = GetState();
    SetState(entry);
    Stmt else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    SetState(Join(std::move(then_state), GetState()));
    Stmt rewritten = IfThenElse(op->condition, then_case, else_case, op->span);
    return PrependSync(std::move(rewritten), prefix);
  }

  Stmt RewriteLoop(const Stmt &original_body, const ForNode *for_loop,
                   const WhileNode *while_loop) {
    LoopInfo info = MakeLoopInfo(original_body, for_loop);
    Accesses consumers = CollectLoopAccesses(original_body);
    Accesses header_loads;
    if (for_loop) {
      CollectTileLoads(for_loop->min, &header_loads);
      CollectTileLoads(for_loop->extent, &header_loads);
      if (for_loop->step.has_value()) {
        CollectTileLoads(for_loop->step.value(), &header_loads);
      }
    } else {
      CollectTileLoads(while_loop->condition, &header_loads);
      MergeAccesses(&consumers, header_loads);
    }

    Accesses covered_consumers;
    for (const UnitAccess &access : consumers) {
      covered_consumers.push_back(CoverLoopAccess(access, info));
    }
    // Only work launched before the loop can be waited here. Header loads are
    // recorded afterwards because their evaluation still belongs to the loop.
    SunmmioSyncUnits prefix = SynchronizeBeforeLoads(header_loads);
    prefix |= SynchronizeBeforeLoads(covered_consumers);
    RecordAccesses(header_loads);
    FlowState entry = GetState();
    loop_stack_.push_back({{{}, false}, {}, active_allocations_.size()});
    Stmt body = VisitStmt(original_body);
    FlowState normal = GetState();
    LoopFrame exits = std::move(loop_stack_.back());
    loop_stack_.pop_back();

    SunmmioSyncUnits normal_units = CarriedSync(normal, consumers, info);
    if (normal.reachable) {
      body = AppendSync(std::move(body), normal_units);
      CompleteStateUnits(&normal, normal_units);
    }
    FlowState outgoing = Join(entry, CoverLoopState(normal, info));
    outgoing = Join(std::move(outgoing), CoverLoopState(exits.breaks, info));
    std::unordered_map<const EvaluateNode *, SunmmioSyncUnits> continue_waits;
    for (auto &[jump, state] : exits.continues) {
      SunmmioSyncUnits units = CarriedSync(state, consumers, info);
      continue_waits.emplace(jump, units);
      CompleteStateUnits(&state, units);
      outgoing = Join(std::move(outgoing), CoverLoopState(state, info));
    }
    body = AddContinueSyncs(std::move(body), continue_waits);
    SetState(outgoing);
    Stmt loop =
        for_loop
            ? Stmt(For(for_loop->loop_var, for_loop->min, for_loop->extent,
                       for_loop->kind, body, for_loop->thread_binding,
                       for_loop->annotations, for_loop->step, for_loop->span))
            : Stmt(While(while_loop->condition, body, while_loop->span));
    return PrependSync(std::move(loop), prefix);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (is_zero(op->extent)) {
      Accesses header_loads;
      CollectTileLoads(op->min, &header_loads);
      if (op->step.has_value()) {
        CollectTileLoads(op->step.value(), &header_loads);
      }
      return PrependSync(ffi::GetRef<Stmt>(op), ProcessAccesses(header_loads));
    }
    return RewriteLoop(op->body, op, nullptr);
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    return RewriteLoop(op->body, nullptr, op);
  }

  std::unordered_map<const BufferStoreNode *, Region> store_regions_;
  Map<Buffer, Layout> layouts_;
  SunmmioTileProcessorConfig config_;
  std::vector<UnitAccess> pending_;
  bool reachable_{true};
  std::vector<LoopFrame> loop_stack_;
  std::vector<Var> active_allocations_;
};

class InjectBroadcastBarriersRewriter : public StmtMutator {
public:
  InjectBroadcastBarriersRewriter(const Target &target,
                                  arith::Analyzer *analyzer)
      : analyzer_(analyzer) {
    SunmmioMeshConfig mesh = GetSunmmioMeshConfig(target);
    mesh_nrow_ = mesh.nrow;
    mesh_ncol_ = mesh.ncol;
    ICHECK_LE(mesh_nrow_ * mesh_ncol_, 64)
        << "tl.broadcast_ barrier mask supports at most 64 cores";
  }

private:
  PrimExpr LocalMaskBitSet(PrimExpr local_mask, int local_index) const {
    PrimExpr bit = I64Imm(static_cast<int64_t>(uint64_t{1} << local_index));
    return (AsI64(local_mask) & bit) != I64Imm(0);
  }

  std::optional<int64_t>
  TryExpandBroadcastLocalMaskImm(const PrimExpr &local_mask, int direction,
                                 const PrimExpr &src_core) const {
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
                                    const PrimExpr &src_core) const {
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
        PrimExpr bit = CoreBitMask(src_row * ncol + I64Imm(col));
        global_mask = Select(LocalMaskBitSet(local_mask, col),
                             AsI64(global_mask) | AsI64(bit), global_mask);
      }
    } else {
      ICHECK_EQ(direction, 1)
          << "tl.broadcast_ local mask expansion only supports direction 0/1";
      for (int row = 0; row < mesh_nrow_; ++row) {
        PrimExpr bit = CoreBitMask(I64Imm(row * mesh_ncol_) + src_col);
        global_mask = Select(LocalMaskBitSet(local_mask, row),
                             AsI64(global_mask) | AsI64(bit), global_mask);
      }
    }
    return analyzer_ ? analyzer_->Simplify(global_mask) : global_mask;
  }

  PrimExpr GetBroadcastBarrierSrcCore(const CallNode *call) const {
    if (BroadcastCallHasSrcCore(call)) {
      return GetBroadcastSrcCore(call);
    }
    ICHECK(current_kernel_core_id_.defined())
        << "tl.broadcast_ without optional src_core requires an enclosing "
           "blockIdx.x binding";
    return current_kernel_core_id_;
  }

  BarrierMaskInfo GetBroadcastBarrierMask(const CallNode *call) const {
    const auto *direction_imm =
        call->args[kBroadcastArgDirection].as<IntImmNode>();
    ICHECK(direction_imm &&
           (direction_imm->value == 0 || direction_imm->value == 1))
        << "tl.broadcast_ barrier mask expansion only supports row or column "
           "broadcasts";
    int direction = static_cast<int>(direction_imm->value);
    PrimExpr src_core = GetBroadcastBarrierSrcCore(call);
    PrimExpr write_mask = ExpandBroadcastLocalMask(
        call->args[kBroadcastArgMask], direction, src_core);

    BarrierMaskInfo info;
    info.expr = AsI64(CoreBitMask(src_core)) | AsI64(write_mask);
    if (analyzer_) {
      info.expr = analyzer_->Simplify(info.expr);
    }
    if (!info.expr.as<IntImmNode>()) {
      info.candidates = EnumerateMaskCandidates(
          info.expr, direction, mesh_nrow_, mesh_ncol_, analyzer_);
      ICHECK(!info.candidates.empty())
          << "Could not derive static candidate masks for dynamic "
             "tl.broadcast_ barrier mask";
    }
    return info;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(broadcast_())) {
      return StmtMutator::VisitStmt_(op);
    }
    BarrierMaskInfo participant_mask = GetBroadcastBarrierMask(call);
    Array<Stmt> stmts;
    stmts.push_back(Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                                  MakeBarrierArgs(participant_mask))));
    stmts.push_back(StmtMutator::VisitStmt_(op));
    stmts.push_back(Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                                  MakeBarrierArgs(participant_mask))));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    PrimExpr old_kernel_core_id = current_kernel_core_id_;
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "blockIdx.x") {
        current_kernel_core_id_ = iv->var;
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    current_kernel_core_id_ = old_kernel_core_id;
    return stmt;
  }

  int mesh_nrow_{0};
  int mesh_ncol_{0};
  arith::Analyzer *analyzer_;
  PrimExpr current_kernel_core_id_;
};

class InitReusableBarriersRewriter : public StmtMutator {
public:
  Stmt operator()(Stmt body) {
    Stmt rewritten = VisitStmt(body);
    return HasThreadExtent(rewritten) ? rewritten
                                      : PrependBarrierInits(rewritten, {});
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
      if (const auto *call = op->value.as<CallNode>()) {
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
      stmts.push_back(Evaluate(
          Call(DataType::Handle(), barrier_init(), MakeBarrierInitArgs(mask))));
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
      return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
    }
    return AttrStmt(op->node, op->attr_key, op->value,
                    PrependBarrierInits(body, scoped_vars_), op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    scoped_vars_.push_back(op->loop_var);
    Stmt body = StmtMutator::VisitStmt(op->body);
    scoped_vars_.pop_back();
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations, op->step, op->span);
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

PrimFunc RewriteSunmmioSync(PrimFunc func, arith::Analyzer *analyzer) {
  Target target = func->GetAttr<Target>(tvm::attr::kTarget).value();
  InjectBroadcastBarriersRewriter inject_barriers(target, analyzer);
  func.CopyOnWrite()->body = inject_barriers(func->body);
  InjectUnitSyncRewriter inject_unit_sync(func);
  func.CopyOnWrite()->body = inject_unit_sync(func->body);
  InitReusableBarriersRewriter init_barriers;
  func.CopyOnWrite()->body = init_barriers(func->body);
  return func;
}

} // namespace

tvm::transform::Pass InjectSunmmioSync() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const PassContext &) -> PrimFunc {
    if (!func->HasNonzeroAttr(tir::attr::kIsGlobalFunc)) {
      return func;
    }
    arith::Analyzer analyzer;
    return RewriteSunmmioSync(std::move(func), &analyzer);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSunmmioSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectSunmmioSync", InjectSunmmioSync);
}

} // namespace tl
} // namespace tvm
