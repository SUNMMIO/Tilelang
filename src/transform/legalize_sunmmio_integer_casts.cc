/*!
 * \file tl/transform/legalize_sunmmio_integer_casts.cc
 * \brief Preserve finite-width scalar casts before generic arithmetic analysis.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../layout/cute_layout.h"
#include "../support/finite_width_integer.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {
namespace {
using namespace tir;

bool IsInteger(DataType t) { return t.is_int() || t.is_uint() || t.is_bool(); }

// The generic analyzer may intersect an unconverted source range with the
// destination type. Leave a numeric Cast only when it preserves its operand.
PrimExpr ProtectCast(const Cast &op, FiniteWidthInteger *facts = nullptr) {
  const PrimExpr &value = op->value;
  DataType src = value.dtype(), dst = op->dtype;
  if (!IsInteger(src) || !IsInteger(dst) || src == dst)
    return op;
  if (dst.is_bool())
    return value != make_zero(src);
  if ((src.is_bool() && (dst.is_uint() || dst.bits() > 1)) ||
      (src.code() == dst.code() && src.bits() <= dst.bits()) ||
      (src.is_uint() && dst.is_int() && src.bits() < dst.bits()))
    return op;

  if (src.is_scalar() && src.bits() <= 64 && dst.bits() <= 64) {
    if (facts && facts->FitsIn(value, dst))
      return op;
    if (auto literal = FiniteWidthInteger::Literal(op))
      return dst.is_uint() ? make_const(dst, static_cast<uint64_t>(*literal))
                           : make_const(dst, static_cast<int64_t>(*literal));
  }
  ICHECK(src.is_scalar() && src.bits() <= 64 &&
         (dst.bits() == 8 || dst.bits() == 16 || dst.bits() == 32 ||
          dst.bits() == 64))
      << "SunMMIO inter-rank cannot preserve integer Cast from " << src
      << " to " << dst << " for " << value
      << "; use a supported scalar integer conversion";

  if (src.bits() <= dst.bits()) {
    // The remaining equal-width or widening conversion changes signedness.
    // Extend in the source signedness, then keep all bits. This also works
    // for 64 bits, whose period cannot be represented by a scalar TIR literal.
    return reinterpret(dst, cast(src.with_bits(dst.bits()), value));
  }

  DataType work = DataType::Int(64);
  int64_t period = int64_t{1} << dst.bits();
  PrimExpr modulus = make_const(work, period);
  PrimExpr half = make_const(work, period / 2);
  // Reduce a 64-bit input before converting it to signed or adding the bias.
  // Every intermediate fits int64, and the original operand occurs only once.
  PrimExpr x = src.bits() == 64 ? value : cast(work, value);
  if (src.bits() > 32) {
    x = cast(work, floormod(x, make_const(x.dtype(), period)));
  }
  PrimExpr result = x;
  if (dst.is_int())
    result = floormod(x + half, modulus) - half;
  else if (src.bits() <= 32)
    result = floormod(x, modulus);
  return Cast(dst, result, op->span);
}

class LegalizeCasts : public StmtExprMutator {
public:
  PrimExpr SimplifyFrontend(const PrimExpr &expr, Optional<Var> rank,
                            int64_t world_size, bool allocation_shape) {
    if (rank)
      analyzer_.Bind(*rank, Range::FromMinExtent(0, Integer(world_size)));
    PrimExpr protected_expr = VisitExpr(expr);
    PrimExpr result = analyzer_.Simplify(protected_expr);
    if (allocation_shape && !protected_expr.same_as(expr)) {
      ICHECK(result.as<IntImmNode>())
          << "SunMMIO inter-rank integer Cast in buffer or annotation metadata "
             "requires a value-preserving conversion: "
          << expr;
    }
    return result;
  }

  Stmt Run(const PrimFunc &func) {
    // T.dist.world_size() is also legal without a RankId parameter.
    if (func->GetAttr<Integer>("tl.dist.rank_id_param_index")) {
      auto context = dist_transform::GetDistPassContext(func).value();
      analyzer_.Bind(context.rank_id,
                     Range::FromMinExtent(0, Integer(context.world_size)));
    }
    CheckMetadata(func->buffer_map);
    CheckMetadata(func->attrs->dict);
    return VisitStmt(func->body);
  }

private:
  // Shapes, explicit layouts and attributes are shared as object identities.
  // If they require a wrapping conversion, fail before any pass uses them for
  // address arithmetic. Normal dynamic local initializers are BufferStores.
  void CheckMetadata(const ffi::Any &object) {
    if (auto expr = object.as<PrimExpr>()) {
      ICHECK(VisitExpr(*expr).same_as(*expr))
          << "SunMMIO inter-rank integer Cast in buffer or annotation metadata "
             "requires a value-preserving conversion: "
          << *expr;
    } else if (auto range = object.as<Range>()) {
      CheckMetadata((*range)->min);
      CheckMetadata((*range)->extent);
    } else if (auto iv = object.as<IterVar>()) {
      CheckMetadata((*iv)->dom);
    } else if (auto buffer = object.as<Buffer>()) {
      if (!checked_buffers_.insert(buffer->get()).second)
        return;
      CheckMetadata((*buffer)->shape);
      CheckMetadata((*buffer)->strides);
      CheckMetadata((*buffer)->elem_offset);
    } else if (auto layout = object.as<Layout>()) {
      CheckMetadata((*layout)->InputShape());
      CheckMetadata((*layout)->GetForwardIndex());
      if (auto cute = (*layout).as<CuteLayoutNode>()) {
        CheckMetadata(cute->GetModeShape());
        CheckMetadata(cute->GetModeStride());
        CheckMetadata(cute->GetLogicalShape());
      }
      if (auto fragment = (*layout).as<FragmentNode>()) {
        CheckMetadata(fragment->GetForwardThread());
        CheckMetadata(fragment->ReplicateExtent());
      }
    } else if (auto array = object.as<Array<ffi::Any>>()) {
      for (const auto &item : *array)
        CheckMetadata(item);
    } else if (auto map = object.as<Map<ffi::Any, ffi::Any>>()) {
      for (const auto &[key, value] : *map) {
        CheckMetadata(key);
        CheckMetadata(value);
      }
    }
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    CheckMetadata(op->annotations);
    return StmtExprMutator::VisitExpr_(op);
  }
  Stmt VisitStmt_(const BlockNode *op) final {
    CheckMetadata(op->alloc_buffers);
    CheckMetadata(op->annotations);
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const AllocateNode *op) final {
    CheckMetadata(op->annotations);
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    CheckMetadata(op->node);
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->thread_binding)
      CheckMetadata(*op->thread_binding);
    CheckMetadata(op->annotations);
    // The frontend represents stepped ranges with a zero-based trip counter.
    // A positive literal extent bounds this SSA variable without evaluating a
    // dynamic endpoint or assuming that an addition fits its type.
    if (const auto *extent = op->extent.as<IntImmNode>();
        extent && extent->value > 0 && is_zero(op->min) &&
        (!op->step || is_one(*op->step))) {
      analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const DeclBufferNode *op) final {
    CheckMetadata(op->buffer);
    return StmtExprMutator::VisitStmt_(op);
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    CheckMetadata(op->buffer);
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (op->predicate) {
      auto predicate = VisitExpr(*op->predicate);
      if (!predicate.same_as(*op->predicate))
        load.CopyOnWrite()->predicate = predicate;
    }
    return load;
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    CheckMetadata(op->buffer);
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (op->predicate) {
      auto predicate = VisitExpr(*op->predicate);
      if (!predicate.same_as(*op->predicate))
        store.CopyOnWrite()->predicate = predicate;
    }
    return store;
  }
  PrimExpr VisitExpr_(const CastNode *op) final {
    auto cast = Downcast<Cast>(StmtExprMutator::VisitExpr_(op));
    FiniteWidthInteger facts(&analyzer_, 1, &defs_);
    return ProtectCast(cast, &facts);
  }
  Stmt VisitStmt_(const LetStmtNode *op) final {
    auto value = VisitExpr(op->value);
    defs_.emplace(op->var.get(), value);
    auto body = VisitStmt(op->body);
    defs_.erase(op->var.get());
    return value.same_as(op->value) && body.same_as(op->body)
               ? ffi::GetRef<Stmt>(op)
               : LetStmt(op->var, value, body, op->span);
  }
  std::unordered_set<const BufferNode *> checked_buffers_;
  arith::Analyzer analyzer_;
  FiniteWidthInteger::Definitions defs_;
};
} // namespace

namespace dist_transform {
PrimExpr CastDistStorage(DataType dtype, const PrimExpr &value) {
  return dtype == value.dtype() ? value : ProtectCast(tir::Cast(dtype, value));
}
} // namespace dist_transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.transform.SimplifySunmmioFrontendInteger",
      [](const PrimExpr &expr, Optional<tir::Var> rank, int64_t world_size,
         bool allocation_shape) {
        return LegalizeCasts().SimplifyFrontend(expr, rank, world_size,
                                                allocation_shape);
      });
  ffi::reflection::GlobalDef().def(
      "tl.transform.LegalizeSunmmioIntegerCasts", []() {
        return tir::transform::CreatePrimFuncPass(
            [](tir::PrimFunc func, const IRModule &,
               const tvm::transform::PassContext &) {
              auto target = func->GetAttr<Target>(tvm::attr::kTarget);
              if (dist_transform::IsMultiRank(func) && target &&
                  TargetIsSunmmio(*target)) {
                auto body = LegalizeCasts().Run(func);
                if (!body.same_as(func->body))
                  func.CopyOnWrite()->body = body;
              }
              return func;
            },
            0, "tl.LegalizeSunmmioIntegerCasts", {});
      });
}
} // namespace tl
} // namespace tvm
