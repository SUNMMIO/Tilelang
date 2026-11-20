/*!
 * \file tl/ir.cc
 * \brief Extension for the tvm script frontend.
 *
 */

#include "./transform/common/attr.h"
#include "op/builtin.h"
#include "tvm/ffi/any.h"
#include <tvm/ffi/object.h>

#include "support/ffi_aliases.h"
#include "support/finite_width_integer.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/ir_builder/tir/ir.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace script::ir_builder::tir;

static Var CreateEnvThread(String name, String thread_tag, DataType dtype) {
  using namespace tvm::tir;
  using namespace tvm::script::ir_builder;
  IterVar iter_var(Range{nullptr}, Var(std::move(name), dtype),
                   tvm::tir::IterVarType::kThreadIndex, std::move(thread_tag));
  Var var = iter_var->var;
  if (Optional<PrimFuncFrame> opt_frame =
          IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    opt_frame.value()->env_threads.Set(var, iter_var);
  } else {
    LOG(FATAL) << "EnvThread can only be used inside a PrimFunc";
  }
  return var;
}

static ForFrame MakeIterVarFrame(const std::string &name, const PrimExpr &dom) {
  using namespace tvm::tir;
  Var var = Var(name, dom->dtype);
  // Create a frame that represents a loop over the given domain.
  ObjectPtr<ForFrameNode> n = tvm::ffi::make_object<ForFrameNode>();
  n->vars.push_back(var);
  n->doms.push_back(Range(0, dom));
  n->f_make_for_loop = [](const Array<Var> &vars, const Array<Range> &doms,
                          const Array<Optional<PrimExpr>> &steps,
                          Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    Optional<PrimExpr> step =
        !steps.empty() ? steps[0] : Optional<PrimExpr>(std::nullopt);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, body,
               /*thread_binding=*/std::nullopt,
               /*annotations=*/tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>{},
               /*step=*/step);
  };
  return ForFrame(n);
}

// Walk outward through lexical Let frames once. Only literal bindings
// are used; function inputs must retain their runtime identity.
static PrimExpr ResolveLoopAlias(PrimExpr expr) {
  using tvm::script::ir_builder::IRBuilder;
  if (!expr.as<tir::VarNode>() || !IRBuilder::IsInScope())
    return expr;
  const auto &frames = IRBuilder::Current()->frames;
  const auto *func =
      frames.empty() ? nullptr : frames[0].as<PrimFuncFrameNode>();
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    const auto *binding = (*it).as<LetFrameNode>();
    if (!binding || !binding->var.same_as(expr))
      continue;
    if (func && std::any_of(func->args.begin(), func->args.end(),
                            [&](const Var &v) { return v.same_as(expr); }))
      return expr;
    expr = binding->value;
    if (!expr.as<tir::VarNode>())
      break;
  }
  return expr;
}

static std::optional<FiniteWidthInteger::Wide>
LoopLiteral(const PrimExpr &expr) {
  return FiniteWidthInteger::Literal(ResolveLoopAlias(expr));
}

// The block axis introduced by axis.remap initially matches the DSL IV.
// Give that axis the full logical range if a narrow IV cannot represent
// its extent. Uses in the block retain their original integer semantics.
class WidenRemappedAxis : public tir::StmtMutator {
public:
  WidenRemappedAxis(Range parsed, PrimExpr start, PrimExpr count,
                    PrimExpr value)
      : parsed_(parsed),
        domain_(Range::FromMinExtent(cast(count.dtype(), start), count)),
        value_(value) {}

private:
  Stmt VisitStmt_(const tir::BlockRealizeNode *op) final {
    auto result = Downcast<tir::BlockRealize>(StmtMutator::VisitStmt_(op));
    auto block = result->block;
    auto bindings = result->iter_values;
    Map<Var, PrimExpr> replacements;
    for (size_t i = 0; i < block->iter_vars.size(); ++i) {
      auto axis = block->iter_vars[i];
      if (!axis->dom.same_as(parsed_))
        continue;
      Var wide(axis->var->name_hint, domain_->extent.dtype());
      replacements.Set(axis->var, cast(axis->var.dtype(), wide));
      block.CopyOnWrite()->iter_vars.Set(
          i, IterVar(domain_, wide, axis->iter_type, axis->thread_tag));
      bindings.Set(i, value_);
    }
    if (replacements.empty())
      return result;
    return tir::BlockRealize(
        bindings, result->predicate,
        Downcast<tir::Block>(Substitute(block, replacements)), result->span);
  }
  Range parsed_, domain_;
  PrimExpr value_;
};

// Keep the original range operands until the entry snapshots and ordering test
// are built. A narrow stop-start may already have lost the empty-range case.
ForFrame NormalizedFor(PrimExpr start, PrimExpr stop, PrimExpr step,
                       bool unroll, bool stepped,
                       const Map<String, tvm::ffi::Any> &annotations) {
  using namespace tvm::tir;
  using Wide = FiniteWidthInteger::Wide;
  for (const auto &operand : {start, stop, step})
    ICHECK((operand.dtype().is_int() || operand.dtype().is_uint()) &&
           operand.dtype().is_scalar() && operand.dtype().bits() <= 64)
        << "serial/unroll bounds and step must be scalar integers up to 64 "
           "bits";
  auto promote = [](DataType a, DataType b) {
    return (make_zero(a) + make_zero(b)).dtype();
  };
  DataType bounds_type = promote(start.dtype(), stop.dtype());
  start = cast(bounds_type, start);
  stop = cast(bounds_type, stop);
  auto step_value = LoopLiteral(step);
  ICHECK(!step_value || *step_value != 0)
      << "Invalid stepped serial: step must be non-zero";
  bool reverse = step_value && *step_value < 0;
  Wide magnitude = step_value ? (reverse ? -*step_value : *step_value) : 0;
  // Match the legacy promotion through a Python integer constant in ceildiv.
  DataType count_type = step.dtype();
  if (step_value)
    count_type = magnitude <= INT32_MAX   ? DataType::Int(32)
                 : magnitude <= INT64_MAX ? DataType::Int(64)
                                          : DataType::UInt(64);
  count_type = promote(bounds_type, count_type);
  DataType iv_type =
      stepped ? promote(bounds_type, promote(count_type, step.dtype()))
              : bounds_type;
  const bool signed_unit = bounds_type.is_int() && step_value &&
                           *step_value == 1 && iv_type == bounds_type;
  // A zero origin cannot overflow stop - start, and the native For
  // evaluates its sole nonconstant bound once on entry.
  if (signed_unit && is_zero(start))
    return unroll
               ? tvm::script::ir_builder::tir::Unroll(start, stop, annotations)
               : tvm::script::ir_builder::tir::Serial(start, stop, annotations);
  DataType work_type = bounds_type.bits() <= 32 &&
                               (!step_value || magnitude <= INT64_MAX) &&
                               step.dtype().bits() <= 32
                           ? DataType::Int(64)
                           : DataType::UInt(64);
  auto a_value = LoopLiteral(start), b_value = LoopLiteral(stop);
  std::vector<std::pair<Var, PrimExpr>> snapshots;
  auto snapshot = [&](const PrimExpr &expr, const char *name,
                      std::optional<Wide> literal) -> PrimExpr {
    if (expr.as<VarNode>() || literal)
      return expr;
    Var saved(name, expr.dtype());
    snapshots.emplace_back(saved, expr);
    return saved;
  };
  PrimExpr a = snapshot(start, "range_start", a_value);
  PrimExpr b = snapshot(stop, "range_stop", b_value);
  PrimExpr s = snapshot(step, "range_step", step_value);
  PrimExpr zero = make_zero(work_type);
  PrimExpr positive_step =
      step_value ? make_const(work_type, static_cast<uint64_t>(magnitude))
                 : cast(work_type, s);
  PrimExpr count;
  Optional<PrimExpr> unroll_guard;
  std::optional<Wide> maximum_count;
  bool use_signed_counter = false;
  if (a_value && b_value && step_value) {
    Wide distance = reverse ? *a_value - *b_value : *b_value - *a_value;
    Wide trips =
        distance <= 0 ? 0 : distance / magnitude + (distance % magnitude != 0);
    count = make_const(work_type, static_cast<uint64_t>(trips));
    maximum_count = trips;
  } else {
    PrimExpr compare_a = a, compare_b = b;
    if (bounds_type.is_uint()) {
      // A sign-bit permutation preserves unsigned order and keeps symbolic
      // endpoint wrap out of mathematical inequality rewrites.
      auto signed_type = DataType::Int(bounds_type.bits());
      auto mask = make_const(
          signed_type,
          static_cast<int64_t>(-(Wide(1) << (bounds_type.bits() - 1))));
      auto ordered_bits = [&](PrimExpr operand, std::optional<Wide> literal) {
        if (literal)
          return make_const(
              signed_type,
              static_cast<int64_t>(*literal -
                                   (Wide(1) << (bounds_type.bits() - 1))));
        return bitwise_xor(cast(signed_type, operand), mask);
      };
      compare_a = ordered_bits(a, a_value);
      compare_b = ordered_bits(b, b_value);
    }
    PrimExpr ordered = reverse ? compare_a > compare_b : compare_a < compare_b;
    auto stable = [](const PrimExpr &expr) {
      if (SideEffect(expr) <= CallEffectKind::kPure)
        return true;
      const auto *load = expr.as<BufferLoadNode>();
      return load && load->buffer.scope() == "local.var";
    };
    auto delta = [&](const PrimExpr &end,
                     const PrimExpr &begin) -> std::optional<Wide> {
      if (!stable(begin))
        return std::nullopt;
      if (StructuralEqual()(end, begin))
        return Wide(0);
      if (const auto *add = end.as<AddNode>()) {
        if (StructuralEqual()(add->a, begin))
          return LoopLiteral(add->b);
        if (StructuralEqual()(add->b, begin))
          return LoopLiteral(add->a);
      }
      return std::nullopt;
    };
    auto span = reverse ? delta(start, stop) : delta(stop, start);
    if (span && *span >= 0 && step_value) {
      Wide trips = *span / magnitude + (*span % magnitude != 0);
      maximum_count = trips;
      auto fixed_count = make_const(work_type, static_cast<uint64_t>(trips));
      count = fixed_count;
      if (trips > 0) {
        // For end = begin + span the positive distance is fixed precisely
        // when the endpoint fits. Testing begin against the last valid
        // start avoids a simplifier folding begin < begin + span to true.
        auto begin = reverse ? stop : start;
        arith::Analyzer analyzer;
        auto bound =
            FiniteWidthInteger(&analyzer).Analyze(ResolveLoopAlias(begin));
        Wide max_begin =
            (Wide(1) << (bounds_type.bits() - bounds_type.is_int())) - 1 -
            *span;
        if (!bound.valid || bound.range.hi > max_begin) {
          if (bounds_type.is_int())
            ordered = (reverse ? b : a) <=
                      make_const(bounds_type, static_cast<int64_t>(max_begin));
          if (unroll)
            unroll_guard = ordered; // UnrollLoop requires a literal extent.
          else
            count = Select(ordered, fixed_count, zero);
        }
      }
    } else {
      // Prove the trip bound on the original typed endpoints. The runtime
      // difference may require unsigned 64-bit arithmetic even when the
      // number of iterations is known to fit a signed index.
      arith::Analyzer analyzer;
      FiniteWidthInteger facts(&analyzer);
      auto lower = facts.Analyze(ResolveLoopAlias(reverse ? stop : start));
      auto upper = facts.Analyze(ResolveLoopAlias(reverse ? start : stop));
      if (lower.valid && upper.valid) {
        Wide max_distance = std::max(Wide(0), upper.range.hi - lower.range.lo);
        // Dynamic steps execute under the positive-entry assertion below.
        Wide min_step = magnitude;
        if (!step_value) {
          auto stride = facts.Analyze(ResolveLoopAlias(step));
          min_step =
              stride.valid ? std::max(Wide(1), stride.range.lo) : Wide(1);
        }
        maximum_count =
            max_distance / min_step + (max_distance % min_step != 0);
        // A generic uint64 distance loses the relation between its two
        // endpoints in the backend analyzer. Carry this proof through the
        // counter dtype rather than asking the backend to rediscover it.
        use_signed_counter =
            bounds_type == DataType::UInt(64) && *maximum_count <= INT64_MAX;
      }
      // Clamp the endpoint before subtraction. Select only consumes immutable
      // entry snapshots, so evaluating its operands eagerly is safe.
      PrimExpr high = reverse ? cast(work_type, a) : cast(work_type, b);
      PrimExpr low = reverse ? cast(work_type, b) : cast(work_type, a);
      PrimExpr distance = Select(ordered, high, low) - low;
      if (step_value && magnitude == 1) {
        count = distance;
      } else {
        PrimExpr q = truncdiv(distance, positive_step);
        PrimExpr r = distance - q * positive_step;
        count = q + cast(work_type, r != zero);
      }
    }
  }
  // Use the source IV directly only when the already ordered count fits.
  bool native_unit =
      (!stepped && bounds_type.is_uint() && !use_signed_counter) ||
      (signed_unit && maximum_count &&
       *maximum_count < (Wide(1) << (iv_type.bits() - 1)));
  PrimExpr native_min = a;
  PrimExpr native_extent = cast(iv_type, count);
  if (native_unit && a_value && *a_value == 0) {
    snapshots.clear();
    native_min = start;
    native_extent = stop;
  }
  DataType counter_type = maximum_count && *maximum_count <= INT64_MAX
                              ? DataType::Int(64)
                              : work_type;
  PrimExpr counter_count = cast(counter_type, count);
  PrimExpr counter_zero = make_zero(counter_type);
  Var logical_iv("v", iv_type), counter("range_index", counter_type);
  ObjectPtr<ForFrameNode> frame = tvm::ffi::make_object<ForFrameNode>();
  frame->vars.push_back(logical_iv);
  // axis.remap queries the logical domain while parsing the body. A
  // narrow signed axis may need a wider extent in the completed block.
  auto logical_domain = Range::FromMinExtent(native_min, native_extent);
  frame->doms.push_back(
      native_unit || signed_unit
          ? logical_domain
          : Range::FromMinExtent(counter_zero, counter_count));
  frame->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                               const Array<Optional<PrimExpr>> &, Stmt body) {
    PrimExpr offset = cast(work_type, counter) * positive_step;
    PrimExpr value =
        reverse ? cast(work_type, a) - offset : cast(work_type, a) + offset;
    if (is_zero(count))
      body = tvm::tir::Evaluate(0);
    else if (native_unit)
      body = tvm::tir::For(vars[0], native_min, native_extent,
                           unroll ? ForKind::kUnrolled : ForKind::kSerial, body,
                           std::nullopt, annotations);
    else {
      if (signed_unit && iv_type.bits() < work_type.bits()) {
        body = WidenRemappedAxis(logical_domain, a, count, value)(body);
      }
      body = Substitute(body, {{vars[0], cast(iv_type, value)}});
      body = tvm::tir::For(counter, counter_zero, counter_count,
                           unroll ? ForKind::kUnrolled : ForKind::kSerial, body,
                           std::nullopt, annotations);
    }
    if (unroll_guard)
      body = tvm::tir::IfThenElse(unroll_guard.value(), body);
    if (!step_value)
      body =
          AssertStmt(s.dtype().is_uint() ? s != make_zero(s.dtype())
                                         : s > make_zero(s.dtype()),
                     StringImm("dynamic serial step must be positive"), body);
    for (auto it = snapshots.rbegin(); it != snapshots.rend(); ++it)
      body = tvm::tir::LetStmt(it->first, it->second, body);
    return body;
  };
  return ForFrame(frame);
}

ForFrame ParallelFor(const Array<PrimExpr> &extents,
                     const Map<String, tvm::ffi::Any> &annotations) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = tvm::ffi::make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto &extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop =
      [annotations](const Array<Var> &vars, const Array<Range> &doms,
                    const Array<Optional<PrimExpr>> &steps, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      Optional<PrimExpr> step =
          i < steps.size() ? steps[i] : Optional<PrimExpr>(std::nullopt);
      // Only attach annotations to the outermost parallel loop.
      // Rationale: In TileLang's design, inner loops cannot govern or annotate
      // their outer loops, while the outermost loop can manage and transform
      // the entire nested region. Placing the layout on the outermost loop
      // lets lowering/validators reason about and rewrite the whole nest.
      // Layout annotations (like parallel_loop_layout) and other hints are
      // read from the outermost loop.
      Map<String, tvm::ffi::Any> loop_annotations;
      if (i == 0) {
        loop_annotations = annotations;
      }
      body = For(var, dom->min, dom->extent, ForKind::kParallel, body,
                 /*thread_binding=*/std::nullopt,
                 /*annotations=*/loop_annotations,
                 /*step=*/step);
    }
    return body;
  };
  return ForFrame(n);
}

ForFrame TilesFor(const Array<PrimExpr> &extents,
                  const Map<String, ObjectRef> &annotations) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = tvm::ffi::make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto &extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", dtype));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop =
      [annotations](const Array<Var> &vars, const Array<Range> &doms,
                    const Array<Optional<PrimExpr>> &steps, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    String tile_domain_key(attr::kTileDomain);
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      Optional<PrimExpr> step =
          i < steps.size() ? steps[i] : Optional<PrimExpr>(std::nullopt);
      Map<String, ObjectRef> loop_annotations;
      for (const auto &kv : annotations) {
        if (i != 0 && kv.first == tile_domain_key) {
          continue;
        }
        loop_annotations.Set(kv.first, kv.second);
      }
      body = For(var, dom->min, dom->extent, ForKind::kSerial, body,
                 /*thread_binding=*/std::nullopt,
                 /*annotations=*/loop_annotations,
                 /*step=*/step);
    }
    return body;
  };
  return ForFrame(n);
}

ForFrame PipelinedFor(PrimExpr start, const PrimExpr &stop, int num_stages,
                      const Array<PrimExpr> &order,
                      const Array<PrimExpr> &stages,
                      const Array<Array<PrimExpr>> &sync,
                      const Array<Array<PrimExpr>> &groups) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = tvm::ffi::make_object<ForFrameNode>();
  DataType dtype = stop.dtype();
  n->vars.push_back(Var("v", dtype));
  n->doms.push_back(Range(std::move(start), stop));
  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           const Array<Optional<PrimExpr>> &steps,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    ICHECK(n == 1);
    Map<String, tvm::ffi::Any> anno;
    if (num_stages > 0)
      anno.Set("num_stages", PrimExpr(num_stages));
    if (!order.empty())
      anno.Set("tl_pipeline_order", order);
    if (!stages.empty())
      anno.Set("tl_pipeline_stage", stages);
    if (!groups.empty())
      anno.Set("tl_pipeline_group", groups);
    Optional<PrimExpr> step =
        !steps.empty() ? steps[0] : Optional<PrimExpr>(std::nullopt);
    body = For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, body,
               /*thread_binding=*/std::nullopt, /*annotations=*/anno,
               /*step=*/step);
    return body;
  };
  return ForFrame(n);
}

ForFrame PersistentFor(const Array<PrimExpr> &domain, const PrimExpr &wave_size,
                       const PrimExpr &index, PrimExpr group_size) {
  using namespace tvm::tir;
  ICHECK(!domain.empty());
  ObjectPtr<ForFrameNode> n = tvm::ffi::make_object<ForFrameNode>();
  n->vars.reserve(domain.size());
  n->doms.reserve(domain.size());
  PrimExpr domain_size = domain[0];
  for (int i = 1; i < domain.size(); i++) {
    domain_size *= domain[i];
  }

  auto waves = ceildiv(domain_size, wave_size);
  auto loop_var = Var("w", waves.dtype());
  group_size = min(group_size, domain[domain.size() - 1]);
  Array<Var> coord_vars;

  for (int i = 0; i < domain.size(); ++i) {
    DataType dtype = domain[i].dtype();
    Var coord("v" + std::to_string(i), dtype);
    coord_vars.push_back(coord);
    n->vars.push_back(coord);
    n->doms.push_back(Range(make_const(dtype, 0), domain[i]));
  }

  Array<PrimExpr> grouped_domain;
  grouped_domain.push_back(truncdiv(domain[domain.size() - 1], group_size));
  for (int i = 0; i < domain.size() - 1; ++i) {
    grouped_domain.push_back(domain[i]);
  }
  grouped_domain.push_back(group_size);

  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           const Array<Optional<PrimExpr>> &steps,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    Map<String, tvm::ffi::Any> anno;
    Array<PrimExpr> idxs(grouped_domain.size(), PrimExpr());
    PrimExpr rem = loop_var * wave_size + index;

    for (int i = grouped_domain.size() - 1; i >= 1; --i) {
      idxs.Set(i, truncmod(rem, grouped_domain[i]));
      rem = truncdiv(rem, grouped_domain[i]);
    }
    idxs.Set(0, rem);

    auto out_if = tvm::tir::IfThenElse(
        domain_size <= (loop_var * wave_size + index),
        tvm::tir::Evaluate(
            tvm::tir::Call(DataType::Handle(), tvm::tl::loop_break(), {})),
        Stmt());

    arith::Analyzer analyzer;
    Stmt new_body = body;
    if (analyzer.CanProveGreaterEqual(waves, 2)) {
      new_body = SeqStmt({out_if, body});
    }
    Optional<PrimExpr> step =
        !steps.empty() ? steps[0] : Optional<PrimExpr>(std::nullopt);
    Stmt outer = For(loop_var, 0, waves, ForKind::kSerial, new_body,
                     /*thread_binding=*/std::nullopt, /*annotations=*/anno,
                     /*step=*/step);
    for (int i = 0; i < vars.size() - 1; ++i) {
      outer = tvm::tir::LetStmt(vars[i], idxs[i + 1], outer);
    }
    outer = tvm::tir::LetStmt(vars[vars.size() - 1],
                              idxs[0] * group_size + idxs[vars.size()], outer);
    return outer;
  };

  return ForFrame(n);
}

/*!
 * \brief A frame that represents a kernel launch.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<KernelLaunchFrameNode>().def_ro(
        "frames", &KernelLaunchFrameNode::frames);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.KernelLaunchFrame",
                                    KernelLaunchFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

/*!
 * \brief Managed reference to KernelLaunchFrameNode.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrame : public TIRFrame {
public:
  explicit KernelLaunchFrame(ObjectPtr<KernelLaunchFrameNode> data)
      : TIRFrame(::tvm::ffi::UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(KernelLaunchFrame, TIRFrame,
                                                KernelLaunchFrameNode);
};

KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                               const Optional<Array<PrimExpr>> &block_size_opt,
                               const Map<String, ffi::Any> &attrs) {
  ObjectPtr<KernelLaunchFrameNode> n =
      tvm::ffi::make_object<KernelLaunchFrameNode>();

  // If the kernel is a CPU kernel, we don't need to launch any threads.
  bool is_cpu_kernel_frame =
      attrs.defined() && attrs.count(tilelang_is_cpu_kernel_frame);

  auto block_size = block_size_opt.value_or(Array<PrimExpr>());

  if (is_cpu_kernel_frame) {
    // Launch CPU Kernel
    ICHECK(grid_size.size() >= 0);
    ICHECK(block_size.empty()) << "CPU kernel cannot have block size";
    ICHECK(attrs.defined());
    // create grid loop var
    for (int i = 0; i < grid_size.size(); i++) {
      n->frames.push_back(
          MakeIterVarFrame("block_var_" + std::to_string(i), grid_size[i]));
    }
  } else {
    // Launch GPU Kernel
    ICHECK(grid_size.size() <= 3);
    if (!grid_size.empty())
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bx", "blockIdx.x", grid_size[0].dtype()),
          grid_size[0]));
    if (grid_size.size() > 1)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("by", "blockIdx.y", grid_size[1].dtype()),
          grid_size[1]));
    if (grid_size.size() > 2)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bz", "blockIdx.z", grid_size[2].dtype()),
          grid_size[2]));
    if (block_size.defined()) {
      ICHECK(block_size.size() <= 3);
      if (!block_size.empty()) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tx", "threadIdx.x", block_size[0].dtype()),
            block_size[0]));
      }
      if (block_size.size() > 1) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("ty", "threadIdx.y", block_size[1].dtype()),
            block_size[1]));
      }
      if (block_size.size() > 2) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tz", "threadIdx.z", block_size[2].dtype()),
            block_size[2]));
      }
    }
  }

  if (attrs.defined()) {
    auto empty_block = tvm::script::ir_builder::tir::Block(MainBlockName);
    empty_block->annotations = attrs;
    n->frames.push_back(empty_block);
  } else {
    n->frames.push_back(tvm::script::ir_builder::tir::Block(MainBlockName));
  }

  return KernelLaunchFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.Parallel", ParallelFor)
      .def("tl.NormalizedFor", NormalizedFor)
      .def("tl.Tiles", TilesFor)
      .def("tl.Pipelined", PipelinedFor)
      .def("tl.Persistent", PersistentFor)
      .def("tl.KernelLaunch", KernelLaunch);
}

class WarpSpecializeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WarpSpecializeFrameNode>().def_ro(
        "frames", &WarpSpecializeFrameNode::frames);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.WarpSpecializeFrame",
                                    WarpSpecializeFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

class WarpSpecializeFrame : public TIRFrame {
public:
  explicit WarpSpecializeFrame(ObjectPtr<WarpSpecializeFrameNode> data)
      : TIRFrame(::tvm::ffi::UnsafeInit{}) {
    ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WarpSpecializeFrame, TIRFrame,
                                                WarpSpecializeFrameNode);
};

WarpSpecializeFrame WarpSpecialize(const Array<IntImm> &warp_group_ids,
                                   const PrimExpr &thread_idx,
                                   int warp_group_size = 128) {
  ObjectPtr<WarpSpecializeFrameNode> n =
      tvm::ffi::make_object<WarpSpecializeFrameNode>();
  PrimExpr condition;
  std::vector<int> warp_groups;
  warp_groups.reserve(warp_group_ids.size());
  for (int i = 0; i < warp_group_ids.size(); i++) {
    warp_groups.push_back(Downcast<IntImm>(warp_group_ids[i])->value);
  }
  std::sort(warp_groups.begin(), warp_groups.end());

  // Merge consecutive groups
  std::vector<std::pair<int, int>> merged;
  for (int group : warp_groups) {
    if (merged.empty() || group != merged.back().second) {
      merged.emplace_back(group, group + 1);
    } else {
      merged.back().second = group + 1;
    }
  }

  for (const auto &[start, end] : merged) {
    PrimExpr min_bound = IntImm(thread_idx.dtype(), start) * warp_group_size;
    PrimExpr max_bound = IntImm(thread_idx.dtype(), end) * warp_group_size;
    PrimExpr range_cond = (thread_idx >= min_bound) && (thread_idx < max_bound);

    if (condition.defined()) {
      condition = tir::Or(condition, range_cond);
    } else {
      condition = range_cond;
    }
  }
  IfFrame if_frame = If(condition);
  AttrFrame attr_frame = Attr(Integer(0), "warp_specialize", Integer(1));
  n->frames.push_back(if_frame);
  n->frames.push_back(Then());
  n->frames.push_back(attr_frame);
  return WarpSpecializeFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.WarpSpecialize", WarpSpecialize)
      .def("tl.SideEffect", SideEffect);
  KernelLaunchFrameNode::RegisterReflection();
  WarpSpecializeFrameNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
