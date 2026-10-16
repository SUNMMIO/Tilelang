/*!
 * \file tl/transform/validate_dynamic_comm_put.cc
 * \brief Validate dynamic inter-core put routes before tile-op lowering.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "../op/comm.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

class ValidateDynamicCommPutMutator : public StmtExprMutator {
public:
  static PrimFunc Run(PrimFunc func) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsSunmmio(target.value())) {
      return func;
    }
    ValidateDynamicCommPutMutator validator(target.value());
    Stmt body = validator.VisitStmt(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  explicit ValidateDynamicCommPutMutator(Target target) {
    auto mesh = GetSunmmioMeshConfig(target);
    mesh_nrows_ = mesh.nrow;
    mesh_ncols_ = mesh.ncol;
  }

  struct Route {
    int64_t src;
    int64_t dst;
    int direction;
  };

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    Array<PrimExpr> iter_values;
    iter_values.reserve(op->iter_values.size());
    for (const PrimExpr &value : op->iter_values) {
      iter_values.push_back(VisitExpr(value));
    }
    PrimExpr predicate = VisitExpr(op->predicate);
    PrimExpr previous_predicate = execution_predicate_;
    execution_predicate_ =
        analyzer_.Simplify(And(previous_predicate, predicate));
    Block block = Downcast<Block>(VisitStmt(op->block));
    execution_predicate_ = previous_predicate;

    if (iter_values.same_as(op->iter_values) &&
        predicate.same_as(op->predicate) && block.same_as(op->block)) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    return BlockRealize(iter_values, predicate, block, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tir::attr::thread_extent) {
      return StmtExprMutator::VisitStmt_(op);
    }
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag != "blockIdx.x") {
      return StmtExprMutator::VisitStmt_(op);
    }
    Var previous_core = current_core_;
    int64_t previous_extent = core_extent_;
    current_core_ = iv->var;
    const auto *extent = op->value.as<IntImmNode>();
    core_extent_ = extent ? extent->value : -1;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    current_core_ = previous_core;
    core_extent_ = previous_extent;
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    PrimExpr previous_predicate = execution_predicate_;

    execution_predicate_ =
        analyzer_.Simplify(And(previous_predicate, condition));
    Stmt then_case = VisitStmt(op->then_case);

    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      execution_predicate_ =
          analyzer_.Simplify(And(previous_predicate, Not(condition)));
      else_case = VisitStmt(op->else_case.value());
    }
    execution_predicate_ = previous_predicate;

    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        ((!else_case.defined() && !op->else_case.defined()) ||
         (else_case.defined() && op->else_case.defined() &&
          else_case.value().same_as(op->else_case.value())))) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    ++control_depth_;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    --control_depth_;
    return result;
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    ++control_depth_;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    --control_depth_;
    return result;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    Map<Var, PrimExpr> replacements;
    for (const auto &[var, bound] : let_bindings_) {
      replacements.Set(tvm::ffi::GetRef<Var>(var), bound);
    }
    let_bindings_.emplace(op->var.get(), Substitute(value, replacements));
    Stmt body = VisitStmt(op->body);
    let_bindings_.erase(op->var.get());
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    return LetStmt(op->var, value, body, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr visited = StmtExprMutator::VisitExpr_(op);
    const auto *call = visited.as<CallNode>();
    ICHECK(call);
    if (!call->op.same_as(PutOp::Get())) {
      return visited;
    }
    ICHECK_EQ(call->args.size(), 5U);
    if (analyzer_.Simplify(call->args[3]).as<IntImmNode>() &&
        analyzer_.Simplify(call->args[4]).as<IntImmNode>()) {
      return visited;
    }
    ICHECK(current_core_.defined() && core_extent_ > 0 &&
           core_extent_ <= mesh_nrows_ * mesh_ncols_)
        << "Dynamic T.comm.put requires a constant blockIdx.x extent within "
           "the Sunmmio mesh";
    ICHECK_EQ(control_depth_, 0)
        << "Dynamic T.comm.put requires uniform control flow";

    std::vector<Route> routes;
    routes.reserve(core_extent_);
    int direction = -1;
    for (int64_t cid = 0; cid < core_extent_; ++cid) {
      int64_t src = Resolve(call->args[3], cid, "src_core");
      int64_t dst = Resolve(call->args[4], cid, "dst_core");
      ICHECK_GE(src, 0) << "T.comm.put src_core is out of range at cid=" << cid;
      ICHECK_GE(dst, 0) << "T.comm.put dst_core is out of range at cid=" << cid;
      ICHECK_LT(src, core_extent_)
          << "T.comm.put src_core is not an active core at cid=" << cid;
      ICHECK_LT(dst, core_extent_)
          << "T.comm.put dst_core is not an active core at cid=" << cid;
      ICHECK_NE(src, dst) << "T.comm.put requires distinct cores at cid="
                          << cid;
      int route_direction = -1;
      if (src / mesh_ncols_ == dst / mesh_ncols_) {
        route_direction = 0;
      } else if (src % mesh_ncols_ == dst % mesh_ncols_) {
        route_direction = 1;
      }
      ICHECK_GE(route_direction, 0)
          << "Dynamic T.comm.put requires two hops at cid=" << cid << ": "
          << src << " -> " << dst;
      if (direction < 0) {
        direction = route_direction;
      }
      ICHECK_EQ(direction, route_direction)
          << "Dynamic T.comm.put requires one uniform row/column direction";
      routes.push_back({src, dst, route_direction});
    }

    std::vector<bool> active;
    active.reserve(core_extent_);
    for (int64_t cid = 0; cid < core_extent_; ++cid) {
      active.push_back(
          Resolve(execution_predicate_, cid, "execution predicate") != 0);
    }

    // A no-guard put schedule is a one-hop permutation. Its disjoint cycles
    // are the minimal groups whose members must execute the same barriers.
    std::vector<int64_t> sender_for_target(core_extent_, -1);
    for (int64_t cid = 0; cid < core_extent_; ++cid) {
      const Route &route = routes[cid];
      ICHECK_EQ(route.src, cid)
          << "Dynamic T.comm.put without a sender guard requires src_core "
             "to equal cid; got src_core="
          << route.src << " at cid=" << cid;
      ICHECK_EQ(sender_for_target[route.dst], -1)
          << "Dynamic T.comm.put has multiple senders for dst_core="
          << route.dst;
      sender_for_target[route.dst] = cid;
    }
    for (int64_t target = 0; target < core_extent_; ++target) {
      ICHECK_NE(sender_for_target[target], -1)
          << "Dynamic T.comm.put has no sender for dst_core=" << target;
    }

    std::vector<bool> cycle_visited(core_extent_, false);
    std::vector<int64_t> barrier_masks(core_extent_, 0);
    for (int64_t start = 0; start < core_extent_; ++start) {
      if (cycle_visited[start]) {
        continue;
      }
      std::vector<int64_t> cycle;
      int64_t cid = start;
      while (!cycle_visited[cid]) {
        cycle_visited[cid] = true;
        cycle.push_back(cid);
        cid = routes[cid].dst;
      }
      ICHECK_EQ(cid, start)
          << "Dynamic T.comm.put destination mapping must decompose into "
             "disjoint cycles";

      int64_t mask = 0;
      bool cycle_active = active[cycle.front()];
      for (int64_t member : cycle) {
        ICHECK_EQ(active[member], cycle_active)
            << "Dynamic T.comm.put execution predicate differs within the "
               "synchronization group containing cid="
            << start;
        mask = static_cast<int64_t>(static_cast<uint64_t>(mask) |
                                    (uint64_t{1} << member));
      }
      for (int64_t member : cycle) {
        barrier_masks[member] = mask;
      }
    }

    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set(kCommPutDirectionAttr,
                    IntImm(DataType::Int(32), direction));
    annotations.Set(kCommBarrierMaskAttr, MakePerCoreValue(barrier_masks));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  PrimExpr MakePerCoreValue(const std::vector<int64_t> &values) {
    ICHECK(!values.empty());
    // Keep the proof symbolic in cid; barrier lowering enumerates its finite
    // candidate masks and allocates one reusable barrier for each cycle.
    PrimExpr result = IntImm(DataType::Int(64), values.back());
    for (int64_t cid = static_cast<int64_t>(values.size()) - 2; cid >= 0;
         --cid) {
      result = Select(current_core_ == IntImm(current_core_.dtype(), cid),
                      IntImm(DataType::Int(64), values[cid]), result);
    }
    return analyzer_.Simplify(result);
  }

  int64_t Resolve(const PrimExpr &expr, int64_t cid, const char *name) {
    Map<Var, PrimExpr> replacements;
    for (const auto &[var, bound] : let_bindings_) {
      replacements.Set(tvm::ffi::GetRef<Var>(var), bound);
    }
    PrimExpr value = Substitute(expr, replacements);
    value = analyzer_.Simplify(Substitute(
        value, {{current_core_, IntImm(current_core_.dtype(), cid)}}));
    const auto *imm = value.as<IntImmNode>();
    ICHECK(imm) << "Dynamic T.comm.put " << name
                << " must depend only on cid and compile-time constants; "
                   "cannot resolve at cid="
                << cid << ": " << value;
    return imm->value;
  }

  arith::Analyzer analyzer_;
  Var current_core_;
  int64_t core_extent_ = -1;
  int64_t mesh_nrows_ = 0;
  int64_t mesh_ncols_ = 0;
  int control_depth_ = 0;
  PrimExpr execution_predicate_ = Bool(true);
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;
};

} // namespace

tvm::transform::Pass ValidateDynamicCommPutRoutes() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const PassContext &) -> PrimFunc {
    return ValidateDynamicCommPutMutator::Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ValidateDynamicCommPutRoutes",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateDynamicCommPutRoutes",
                        ValidateDynamicCommPutRoutes);
}

} // namespace tl
} // namespace tvm
