#ifndef TVM_TL_SUPPORT_FINITE_WIDTH_INTEGER_H_
#define TVM_TL_SUPPORT_FINITE_WIDTH_INTEGER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <unordered_map>

namespace tvm {
namespace tl {

// A deliberately bounded integer abstraction. Mathematical expressions are
// retained only when their result fits the runtime dtype. A possible wrap is
// represented by a fresh value with a conservative range. Congruences are kept
// separately, so an unsafe mathematical factor can never leak into a proof.
class FiniteWidthInteger {
public:
  using Wide = __int128_t;
  using Definitions = std::unordered_map<const tir::VarNode *, PrimExpr>;
  struct Range {
    Wide lo, hi;
  };
  struct Value {
    PrimExpr expr;
    Range range;
    int64_t modulus{1}, residue{0};
    bool valid{true};
  };

  // Decode literal encodings only. This does not run range analysis or fold
  // expressions containing variables or memory reads.
  static std::optional<Wide> Literal(const PrimExpr &expr) {
    using namespace tir;
    if (const auto *imm = expr.as<IntImmNode>())
      return Wide(imm->value);
    if (const auto *cast = expr.as<CastNode>(); cast && Integer(cast->dtype)) {
      auto value = Literal(cast->value);
      if (value)
        return cast->dtype.is_bool() ? Wide(*value != 0)
                                     : Wrap(*value, cast->dtype);
    }
    if (const auto *call = expr.as<CallNode>();
        call && call->op.same_as(builtin::large_uint_imm()) &&
        call->args.size() == 2) {
      const auto *lo = call->args[0].as<IntImmNode>();
      const auto *hi = call->args[1].as<IntImmNode>();
      if (lo && hi && lo->value >= 0 && hi->value >= 0 &&
          Wide(lo->value) < (Wide(1) << 32) &&
          Wide(hi->value) < (Wide(1) << 32))
        return Wide(lo->value) + (Wide(hi->value) << 32);
    }
    return std::nullopt;
  }

  explicit FiniteWidthInteger(arith::Analyzer *analyzer, int64_t divisor = 1,
                              const Definitions *definitions = nullptr)
      : analyzer_(analyzer), divisor_(divisor), definitions_(definitions) {
    ICHECK_GT(divisor_, 0);
  }

  Value Analyze(const PrimExpr &expr) {
    using namespace tir;
    if (++work_ > 16384 || !Integer(expr.dtype()))
      return Unknown(expr.dtype(), false);
    if (const auto *op = expr.as<IntImmNode>())
      return Constant(op->value, expr.dtype());
    if (const auto *op = expr.as<VarNode>()) {
      auto saved = variables_.find(op);
      if (saved != variables_.end())
        return saved->second;
      if (definitions_) {
        auto definition = definitions_->find(op);
        if (definition != definitions_->end()) {
          // Also bound malformed cyclic definitions without recursive growth.
          variables_.emplace(op, Unknown(expr.dtype()));
          Value value = Analyze(definition->second);
          // Conditions constrain this immutable SSA value, not future reads
          // from the address used to define it. Keep definition congruences,
          // but intersect a scoped value range using an independent snapshot.
          Range narrowed = value.range;
          auto bound = analyzer_->const_int_bound(expr);
          if (bound->min_value != arith::ConstIntBound::kNegInf &&
              bound->min_value != arith::ConstIntBound::kPosInf)
            narrowed.lo = std::max(narrowed.lo, Wide(bound->min_value));
          if (bound->max_value != arith::ConstIntBound::kPosInf)
            narrowed.hi = std::min(narrowed.hi, Wide(bound->max_value));
          if (narrowed.lo > narrowed.hi)
            value = Unknown(expr.dtype(), false);
          else if (narrowed.lo != value.range.lo ||
                   narrowed.hi != value.range.hi)
            value = narrowed.lo == narrowed.hi
                        ? Constant(narrowed.lo, expr.dtype(), value.valid)
                        : Opaque(expr.dtype(), narrowed, value.modulus,
                                 value.residue, value.valid);
          variables_[op] = value;
          return value;
        }
      }
      Range range = TypeRange(expr.dtype());
      auto bound = analyzer_->const_int_bound(expr);
      if (bound->min_value != arith::ConstIntBound::kNegInf &&
          bound->min_value != arith::ConstIntBound::kPosInf)
        range.lo = std::max(range.lo, Wide(bound->min_value));
      if (bound->max_value != arith::ConstIntBound::kPosInf)
        range.hi = std::min(range.hi, Wide(bound->max_value));
      if (range.lo > range.hi)
        return Unknown(expr.dtype());
      if (range.lo == range.hi)
        return Constant(range.lo, expr.dtype());
      return {expr, range};
    }
    if (const auto *op = expr.as<CastNode>()) {
      if (!Integer(op->value.dtype()))
        return Unknown(op->dtype);
      Value a = Analyze(op->value);
      if (op->dtype.is_bool()) {
        if (a.range.lo == 0 && a.range.hi == 0)
          return Constant(0, op->dtype, a.valid);
        if (a.range.lo > 0 || a.range.hi < 0)
          return Constant(1, op->dtype, a.valid);
        if (a.range.lo < 0 || a.range.hi > 1)
          return Unknown(op->dtype, a.valid);
      }
      return Finish(op->dtype, a.range, a.modulus, a.residue, a.valid,
                    [&] { return Cast(op->dtype, a.expr); });
    }
    if (const auto *op = expr.as<SelectNode>())
      return Choose(op->condition, op->true_value, op->false_value);
    if (const auto *op = expr.as<NotNode>()) {
      Value a = Analyze(op->a);
      if (a.range.lo == a.range.hi)
        return Constant(!a.range.lo, op->dtype, a.valid);
      return {Not(a.expr), {0, 1}, 1, 0, a.valid};
    }
    if (const auto *op = expr.as<CallNode>()) {
      if (op->op.same_as(builtin::large_uint_imm())) {
        auto value = Literal(expr);
        return value ? Constant(*value, op->dtype) : Unknown(op->dtype, false);
      }
      if (op->op.same_as(builtin::if_then_else()))
        return Choose(op->args[0], op->args[1], op->args[2]);
      if (op->op.same_as(builtin::likely()))
        return Analyze(op->args[0]);
      if (op->op.same_as(builtin::bitwise_and()) && op->args.size() == 2) {
        Value a = Analyze(op->args[0]), b = Analyze(op->args[1]);
        bool valid = a.valid && b.valid;
        if (a.range.lo == a.range.hi && b.range.lo == b.range.hi)
          return Constant(a.range.lo & b.range.lo, op->dtype, valid);
        // A nonnegative constant mask clears the sign bit as well as every
        // bit above the mask. No assumptions about the other operand are
        // needed.
        const Value *mask = nullptr;
        if (a.range.lo == a.range.hi && a.range.lo >= 0)
          mask = &a;
        else if (b.range.lo == b.range.hi && b.range.lo >= 0)
          mask = &b;
        if (mask) {
          if (mask->range.lo == 0)
            return Constant(0, op->dtype, valid);
          return Opaque(op->dtype, {0, mask->range.lo}, 1, 0, valid);
        }
        return Unknown(op->dtype, valid);
      }
      // Read/state calls have an independent value at each evaluation point.
      return Unknown(op->dtype);
    }
    if (expr.as<BufferLoadNode>() || expr.as<ProducerLoadNode>())
      return Unknown(expr.dtype());

#define TL_FINITE_BINARY(Kind, Name)                                           \
  if (const auto *op = expr.as<Kind##Node>())                                  \
  return Binary(op->a, op->b, op->dtype, Name)
    TL_FINITE_BINARY(Add, "add");
    TL_FINITE_BINARY(Sub, "sub");
    TL_FINITE_BINARY(Mul, "mul");
    TL_FINITE_BINARY(Div, "div");
    TL_FINITE_BINARY(Mod, "mod");
    TL_FINITE_BINARY(FloorDiv, "floor_div");
    TL_FINITE_BINARY(FloorMod, "floor_mod");
    TL_FINITE_BINARY(Min, "min");
    TL_FINITE_BINARY(Max, "max");
    TL_FINITE_BINARY(EQ, "eq");
    TL_FINITE_BINARY(NE, "ne");
    TL_FINITE_BINARY(LT, "lt");
    TL_FINITE_BINARY(LE, "le");
    TL_FINITE_BINARY(GT, "gt");
    TL_FINITE_BINARY(GE, "ge");
    TL_FINITE_BINARY(And, "and");
    TL_FINITE_BINARY(Or, "or");
#undef TL_FINITE_BINARY
    return Unknown(expr.dtype());
  }

  PrimExpr Normalize(const PrimExpr &expr) {
    Value value = Analyze(expr);
    ICHECK(value.valid) << "Cannot prove finite-width integer expression "
                        << expr
                        << " within analysis budget or supported domain";
    return value.expr;
  }

  bool ProvesMultiple(const PrimExpr &expr) {
    return ProvesMultiple(Analyze(expr));
  }

  bool ProvesMultiple(const Value &value) {
    if (!value.valid)
      return false;
    if (value.modulus == divisor_ && value.residue == 0)
      return true;
    Range i64 = TypeRange(DataType::Int(64));
    if (value.range.lo < i64.lo || value.range.hi > i64.hi)
      return false;
    // The model contains only value-preserving operations or opaque snapshots.
    return analyzer_->CanProveEqual(
        floormod(tir::Cast(DataType::Int(64), value.expr),
                 tir::make_const(DataType::Int(64), divisor_)),
        0);
  }

private:
  static bool Integer(DataType dtype) {
    return (dtype.is_int() || dtype.is_uint() || dtype.is_bool()) &&
           dtype.lanes() == 1 && dtype.bits() > 0 && dtype.bits() <= 64;
  }
  static Range TypeRange(DataType dtype) {
    if (dtype.is_bool())
      return {0, 1};
    if (!Integer(dtype))
      return {-(Wide(1) << 63), (Wide(1) << 63) - 1};
    return dtype.is_uint() ? Range{0, (Wide(1) << dtype.bits()) - 1}
                           : Range{-(Wide(1) << (dtype.bits() - 1)),
                                   (Wide(1) << (dtype.bits() - 1)) - 1};
  }
  static Wide Wrap(Wide value, DataType dtype) {
    Wide period = Wide(1) << dtype.bits();
    value %= period;
    if (value < 0)
      value += period;
    if (dtype.is_int() && value >= period / 2)
      value -= period;
    return value;
  }
  static int64_t Mod(Wide value, int64_t modulus) {
    Wide r = value % modulus;
    return static_cast<int64_t>(r < 0 ? r + modulus : r);
  }
  static int64_t GcdPeriod(int64_t modulus, int bits) {
    int64_t factor = 1;
    while (bits-- > 0 && modulus % 2 == 0) {
      modulus /= 2;
      factor *= 2;
    }
    return factor;
  }
  Value Constant(Wide value, DataType dtype, bool valid = true) {
    value = Wrap(value, dtype);
    PrimExpr expr = dtype.is_uint()
                        ? tir::make_const(dtype, static_cast<uint64_t>(value))
                        : tir::make_const(dtype, static_cast<int64_t>(value));
    return {expr, {value, value}, divisor_, Mod(value, divisor_), valid};
  }
  Value Opaque(DataType dtype, Range range, int64_t modulus = 1,
               int64_t residue = 0, bool valid = true) {
    tir::Var symbol("finite_int", dtype);
    if (Integer(dtype)) {
      constexpr int64_t inf = arith::ConstIntBound::kPosInf;
      constexpr int64_t neg_inf = arith::ConstIntBound::kNegInf;
      int64_t lo = range.lo > inf ? 0 : static_cast<int64_t>(range.lo);
      int64_t hi = range.hi >= inf ? inf : static_cast<int64_t>(range.hi);
      analyzer_->const_int_bound.Update(
          symbol, arith::ConstIntBound(std::max(lo, neg_inf), hi));
    }
    return {symbol, range, modulus, residue, valid};
  }
  Value Unknown(DataType dtype, bool valid = true) {
    return Opaque(dtype, TypeRange(dtype), 1, 0, valid);
  }
  template <typename Make>
  Value Finish(DataType dtype, Range range, int64_t modulus, int64_t residue,
               bool valid, Make make) {
    Range type = TypeRange(dtype);
    if (range.lo == range.hi)
      return Constant(range.lo, dtype, valid);
    if (range.lo >= type.lo && range.hi <= type.hi)
      return {make(), range, modulus, residue, valid};
    modulus = GcdPeriod(modulus, dtype.bits());
    return Opaque(dtype, type, modulus, Mod(residue, modulus), valid);
  }

  Value Choose(const PrimExpr &cond, const PrimExpr &yes, const PrimExpr &no) {
    Value c = Analyze(cond);
    if (c.range.lo == c.range.hi) {
      Value chosen = Analyze(c.range.lo ? yes : no);
      chosen.valid &= c.valid;
      return chosen;
    }
    Value a = Analyze(yes), b = Analyze(no);
    int64_t modulus = std::gcd(std::gcd(a.modulus, b.modulus),
                               std::abs(a.residue - b.residue));
    return {
        tir::Select(c.expr, a.expr, b.expr),
        {std::min(a.range.lo, b.range.lo), std::max(a.range.hi, b.range.hi)},
        modulus,
        Mod(a.residue, modulus),
        c.valid && a.valid && b.valid};
  }

  Value Binary(const PrimExpr &lhs, const PrimExpr &rhs, DataType dtype,
               const std::string &op) {
    using namespace tir;
    // Integer control-flow analysis need not understand floating payload
    // predicates. Both outcomes remain possible without inventing a fact.
    if (!Integer(lhs.dtype()) || !Integer(rhs.dtype()))
      return Unknown(dtype);
    Value a = Analyze(lhs), b = Analyze(rhs);
    bool valid = a.valid && b.valid;
    int64_t modulus = std::gcd(a.modulus, b.modulus);
    int64_t residue = Mod(Wide(a.residue) + b.residue, modulus);
    Range range;
    if (op == "add" || op == "sub" || op == "mul") {
      if (op == "sub" && StructuralEqual()(a.expr, b.expr))
        return Constant(0, dtype, valid);
      if (op == "add")
        range = {a.range.lo + b.range.lo, a.range.hi + b.range.hi};
      else if (op == "sub") {
        range = {a.range.lo - b.range.hi, a.range.hi - b.range.lo};
        residue = Mod(Wide(a.residue) - b.residue, modulus);
      } else {
        Wide products[4];
        bool overflow =
            __builtin_mul_overflow(a.range.lo, b.range.lo, &products[0]) ||
            __builtin_mul_overflow(a.range.lo, b.range.hi, &products[1]) ||
            __builtin_mul_overflow(a.range.hi, b.range.lo, &products[2]) ||
            __builtin_mul_overflow(a.range.hi, b.range.hi, &products[3]);
        if (overflow) {
          range = TypeRange(dtype);
          // The product may wrap; the congruence below still applies.
        } else {
          range = {*std::min_element(products, products + 4),
                   *std::max_element(products, products + 4)};
        }
        modulus = std::gcd(
            std::gcd(divisor_, Mod(Wide(a.modulus) * b.modulus, divisor_)),
            std::gcd(Mod(Wide(a.modulus) * b.residue, divisor_),
                     Mod(Wide(b.modulus) * a.residue, divisor_)));
        residue = Mod(Wide(a.residue) * b.residue, modulus);
        // Arithmetic overflow of the host abstraction also requires wrap
        // normalization; no multiplication in a proof is unchecked.
        if (overflow) {
          modulus = GcdPeriod(modulus, dtype.bits());
          return Opaque(dtype, range, modulus, Mod(residue, modulus), valid);
        }
      }
      return Finish(dtype, range, modulus, residue, valid, [&] {
        return op == "add"   ? a.expr + b.expr
               : op == "sub" ? a.expr - b.expr
                             : a.expr * b.expr;
      });
    }
    if (op == "min" || op == "max") {
      auto fn = [&](Wide x, Wide y) {
        return op == "min" ? std::min(x, y) : std::max(x, y);
      };
      range = {fn(a.range.lo, b.range.lo), fn(a.range.hi, b.range.hi)};
      modulus = std::gcd(modulus, std::abs(a.residue - b.residue));
      return {op == "min" ? min(a.expr, b.expr) : max(a.expr, b.expr), range,
              modulus, Mod(a.residue, modulus), valid};
    }
    if (op == "div" || op == "floor_div" || op == "mod" || op == "floor_mod") {
      if (b.range.lo <= 0 && b.range.hi >= 0)
        return Unknown(dtype, false);
      if ((op == "div" || op == "floor_div") && dtype.is_int() &&
          a.range.lo == TypeRange(dtype).lo && b.range.lo <= -1 &&
          b.range.hi >= -1)
        return Unknown(dtype, false);
      auto calculate = [&](Wide x, Wide y) {
        Wide q = x / y, r = x % y;
        if ((op == "floor_div" || op == "floor_mod") && r != 0 &&
            ((r < 0) != (y < 0))) {
          --q;
          r += y;
        }
        return op == "div" || op == "floor_div" ? q : r;
      };
      Wide values[4] = {
          calculate(a.range.lo, b.range.lo), calculate(a.range.lo, b.range.hi),
          calculate(a.range.hi, b.range.lo), calculate(a.range.hi, b.range.hi)};
      if (a.range.lo == a.range.hi && b.range.lo == b.range.hi)
        return Constant(values[0], dtype, valid);
      if (op == "mod" || op == "floor_mod") {
        Wide magnitude = std::max(b.range.hi < 0 ? -b.range.hi : b.range.hi,
                                  b.range.lo < 0 ? -b.range.lo : b.range.lo);
        range = {op == "floor_mod" ? (b.range.lo > 0 ? 0 : -magnitude + 1)
                                   : (a.range.lo >= 0 ? 0 : -magnitude + 1),
                 op == "floor_mod" ? (b.range.hi < 0 ? 0 : magnitude - 1)
                                   : (a.range.hi <= 0 ? 0 : magnitude - 1)};
      } else {
        range = {*std::min_element(values, values + 4),
                 *std::max_element(values, values + 4)};
      }
      PrimExpr model = op == "div"         ? div(a.expr, b.expr)
                       : op == "mod"       ? truncmod(a.expr, b.expr)
                       : op == "floor_div" ? floordiv(a.expr, b.expr)
                                           : floormod(a.expr, b.expr);
      return {model, range, 1, 0, valid};
    }
    PrimExpr model;
    if (op == "eq")
      model = a.expr == b.expr;
    if (op == "ne")
      model = a.expr != b.expr;
    if (op == "lt")
      model = a.expr < b.expr;
    if (op == "le")
      model = a.expr <= b.expr;
    if (op == "gt")
      model = a.expr > b.expr;
    if (op == "ge")
      model = a.expr >= b.expr;
    if (op == "and")
      model = And(a.expr, b.expr);
    if (op == "or")
      model = Or(a.expr, b.expr);
    ICHECK(model.defined());
    model = analyzer_->Simplify(model);
    if (const auto *imm = model.as<IntImmNode>())
      return Constant(imm->value, dtype, valid);
    return {model, {0, 1}, 1, 0, valid};
  }

  arith::Analyzer *analyzer_;
  int64_t divisor_;
  const Definitions *definitions_;
  size_t work_{0};
  std::unordered_map<const tir::VarNode *, Value> variables_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_SUPPORT_FINITE_WIDTH_INTEGER_H_
