/*!
 * \file tl/transform/inject_dist_sync.cc
 * \brief Materialize Rank submit synchronization and signal-state updates.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

// Track the implicit per-core send queue across structured control flow.
class DistQueueValidator : public StmtVisitor {
public:
  struct Result {
    std::unordered_set<const EvaluateNode *> returns_needing_wait;
    std::unordered_set<const AttrStmtNode *> core_exits_needing_wait;
    std::unordered_set<const EvaluateNode *> empty_submits;
  };

  static Result Validate(const Stmt &body) {
    DistQueueValidator validator;
    validator(body);
    if (validator.falls_through_) {
      validator.CheckSubmitted(/*before_return=*/false);
      ICHECK(!validator.channel_in_flight_)
          << "Cannot insert T.dist sender completion without a blockIdx.x core";
    }
    return Result{std::move(validator.returns_needing_wait_),
                  std::move(validator.core_exits_needing_wait_),
                  std::move(validator.empty_submits_)};
  }

private:
  struct State {
    bool queue_pending;
    bool strict_batch_open;
    bool channel_in_flight;
    bool falls_through;
  };

  State GetState() const {
    return State{queue_pending_, strict_batch_open_, channel_in_flight_,
                 falls_through_};
  }

  void SetState(const State &state) {
    queue_pending_ = state.queue_pending;
    strict_batch_open_ = state.strict_batch_open;
    channel_in_flight_ = state.channel_in_flight;
    falls_through_ = state.falls_through;
  }

  static State Join(const State &lhs, const State &rhs) {
    if (!lhs.falls_through) {
      return rhs;
    }
    if (!rhs.falls_through) {
      return lhs;
    }
    ICHECK_EQ(lhs.strict_batch_open, rhs.strict_batch_open);
    return State{lhs.queue_pending || rhs.queue_pending, lhs.strict_batch_open,
                 lhs.channel_in_flight || rhs.channel_in_flight, true};
  }

  void CheckSubmitted(bool before_return) const {
    ICHECK(!strict_batch_open_)
        << "T.dist submit=True batch is missing its submit boundary"
        << (before_return ? " before return" : "");
    ICHECK(!queue_pending_) << "T.dist send queue is not submitted"
                            << (before_return ? " before return" : "")
                            << "; call T.dist.submit() or use submit=True";
  }

  State VisitBranch(const Stmt &stmt, const State &incoming) {
    SetState(incoming);
    VisitStmt(stmt);
    ICHECK(!falls_through_ || strict_batch_open_ == incoming.strict_batch_open)
        << "T.dist submit=True must begin and submit within the same control-"
           "flow branch";
    return GetState();
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    for (const Stmt &stmt : op->seq) {
      if (!falls_through_) {
        break;
      }
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    const auto *thread = op->node.as<IterVarNode>();
    bool compute_core = op->attr_key == tir::attr::thread_extent && thread &&
                        thread->thread_tag == "blockIdx.x";
    if (!compute_core) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    ++core_depth_;
    StmtVisitor::VisitStmt_(op);
    --core_depth_;
    if (falls_through_) {
      CheckSubmitted(/*before_return=*/false);
      if (channel_in_flight_) {
        core_exits_needing_wait_.insert(op);
        // The planned wait completes this core's outstanding sends before
        // control leaves its scope (including an outer function return).
        channel_in_flight_ = false;
      }
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    State incoming = GetState();
    State then_state = VisitBranch(op->then_case, incoming);
    State else_state = incoming;
    if (op->else_case) {
      else_state = VisitBranch(op->else_case.value(), incoming);
    }
    SetState(Join(then_state, else_state));
  }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    State incoming = GetState();
    ++loop_depth_;
    State entry = incoming;
    while (true) {
      State body_state = VisitBranch(op->body, entry);
      State next = Join(incoming, body_state);
      if (next.queue_pending == entry.queue_pending &&
          next.channel_in_flight == entry.channel_in_flight) {
        break;
      }
      entry = next;
    }
    --loop_depth_;
    SetState(entry);
  }

  void VisitStmt_(const WhileNode *op) final {
    VisitExpr(op->condition);
    State incoming = GetState();
    ++loop_depth_;
    State entry = incoming;
    while (true) {
      State body_state = VisitBranch(op->body, entry);
      State next = Join(incoming, body_state);
      if (next.queue_pending == entry.queue_pending &&
          next.channel_in_flight == entry.channel_in_flight) {
        break;
      }
      entry = next;
    }
    --loop_depth_;
    SetState(entry);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    if (call->op.same_as(builtin::ret())) {
      CheckSubmitted(/*before_return=*/true);
      if (channel_in_flight_) {
        ICHECK_GT(core_depth_, 0) << "Cannot insert T.dist sender completion "
                                     "without a blockIdx.x core";
        returns_needing_wait_.insert(op);
      }
      // Record exits independently; later waits and loop backedges only
      // describe paths that continue executing.
      falls_through_ = false;
      return;
    }
    if (call->op.same_as(dist_batch_begin_())) {
      ICHECK(!queue_pending_)
          << "T.dist operation with submit=True requires an empty send queue; "
             "call T.dist.submit() before it";
      ICHECK(!strict_batch_open_) << "Nested T.dist submit=True batch";
      strict_batch_open_ = true;
      return;
    }
    if (call->op.same_as(dist_put_()) || call->op.same_as(dist_signal_put_())) {
      queue_pending_ = true;
      return;
    }
    if (call->op.same_as(dist_submit_())) {
      bool nonempty = queue_pending_;
      if (!nonempty && loop_depth_ == 0) {
        empty_submits_.insert(op);
      }
      queue_pending_ = false;
      strict_batch_open_ = false;
      channel_in_flight_ |= nonempty;
      return;
    }
    if (call->op.same_as(dist_wait_send())) {
      channel_in_flight_ = false;
      return;
    }
    StmtVisitor::VisitStmt_(op);
  }

  bool queue_pending_{false};
  bool strict_batch_open_{false};
  bool channel_in_flight_{false};
  bool falls_through_{true};
  int loop_depth_{0};
  int core_depth_{0};
  std::unordered_set<const EvaluateNode *> returns_needing_wait_;
  std::unordered_set<const AttrStmtNode *> core_exits_needing_wait_;
  std::unordered_set<const EvaluateNode *> empty_submits_;
};

// Temporary TIR cleanup. The final sender drain belongs in codegen's common
// device epilogue; this analysis does not cover every break/continue exit path.
class InjectDistExitWait : public StmtExprMutator {
public:
  static PrimFunc Run(PrimFunc func, const DistQueueValidator::Result &queue) {
    InjectDistExitWait injector(queue);
    Stmt body = injector(func->body);
    func.CopyOnWrite()->body = std::move(body);
    return func;
  }

private:
  explicit InjectDistExitWait(const DistQueueValidator::Result &queue)
      : queue_(queue) {}

  Stmt WaitSend() const {
    return Evaluate(Call(DataType::Handle(), dist_wait_send(), {}));
  }

  Stmt AppendCoreWait(const Stmt &stmt) const {
    if (const auto *alloc = stmt.as<AllocateNode>()) {
      return Allocate(alloc->buffer_var, alloc->dtype, alloc->extents,
                      alloc->condition, AppendCoreWait(alloc->body),
                      alloc->annotations, alloc->span);
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      return DeclBuffer(decl->buffer, AppendCoreWait(decl->body), decl->span);
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return LetStmt(let->var, let->value, AppendCoreWait(let->body),
                     let->span);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      Block block = realize->block;
      block.CopyOnWrite()->body = AppendCoreWait(block->body);
      return BlockRealize(realize->iter_values, realize->predicate, block,
                          realize->span);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> statements = seq->seq;
      statements.Set(statements.size() - 1, AppendCoreWait(statements.back()));
      return SeqStmt::Flatten(statements);
    }
    return SeqStmt::Flatten(Array<Stmt>{stmt, WaitSend()});
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (!queue_.core_exits_needing_wait.count(op)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Stmt body = VisitStmt(op->body);
    body = AppendCoreWait(body);
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (queue_.returns_needing_wait.count(op)) {
      return SeqStmt::Flatten(
          Array<Stmt>{WaitSend(), tvm::ffi::GetRef<Stmt>(op)});
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  const DistQueueValidator::Result &queue_;
};

class InjectDistSignalAdvance : public StmtExprMutator {
public:
  explicit InjectDistSignalAdvance(
      std::unordered_set<const EvaluateNode *> empty_submits)
      : empty_submits_(std::move(empty_submits)) {}

  static PrimFunc
  Run(PrimFunc func,
      std::unordered_set<const EvaluateNode *> empty_submits = {}) {
    InjectDistSignalAdvance injector(std::move(empty_submits));
    Stmt body = injector(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *evaluate_node) final {
    const auto *call = evaluate_node->value.as<CallNode>();
    if (!call) {
      return StmtExprMutator::VisitStmt_(evaluate_node);
    }
    if (call->op.same_as(dist_batch_begin_())) {
      ICHECK(call->args.empty());
      return Evaluate(0);
    }
    if (call->op.same_as(dist_submit_())) {
      ICHECK(call->args.empty());
      if (empty_submits_.count(evaluate_node)) {
        return Evaluate(0);
      }
      Stmt wait_idle = Evaluate(Call(DataType::Handle(), dist_wait_send(), {}));
      return SeqStmt::Flatten(
          Array<Stmt>{wait_idle, tvm::ffi::GetRef<Stmt>(evaluate_node)});
    }
    if (call->op.same_as(dist_put_())) {
      ICHECK_EQ(call->args.size(), 6U)
          << "tl.dist_put_ must have its stable peer signature before "
             "InjectDistSync";
      const DistSignalKindInfo &kind =
          RequireDistSignalKindInfo(call->args[3], "signal kind");
      if (kind.update_mode == DistSignalUpdateMode::kIncrement) {
        return tvm::ffi::GetRef<Stmt>(evaluate_node);
      }
      BufferLoad generation = RequireLocalState(call->args[5], "generation");
      PrimExpr next = Cast(generation->dtype,
                           generation + make_const(generation->dtype, 1));
      Stmt advance = BufferStore(generation->buffer, next, generation->indices);
      return SeqStmt::Flatten(
          Array<Stmt>{advance, tvm::ffi::GetRef<Stmt>(evaluate_node)});
    }
    if (call->op.same_as(dist_signal_put_())) {
      ICHECK_EQ(call->args.size(), 4U);
      const DistSignalKindInfo &kind =
          RequireDistSignalKindInfo(call->args[1], "signal kind");
      if (kind.update_mode == DistSignalUpdateMode::kIncrement) {
        return tvm::ffi::GetRef<Stmt>(evaluate_node);
      }
      BufferLoad generation = RequireLocalState(call->args[3], "generation");
      PrimExpr next = Cast(generation->dtype,
                           generation + make_const(generation->dtype, 1));
      Stmt advance = BufferStore(generation->buffer, next, generation->indices);
      return SeqStmt::Flatten(
          Array<Stmt>{advance, tvm::ffi::GetRef<Stmt>(evaluate_node)});
    }
    if (call->op.same_as(dist_expect_())) {
      ICHECK_EQ(call->args.size(), 2U);
      BufferLoad expected = RequireLocalState(call->args[0], "expected");
      PrimExpr delta = Cast(expected->dtype, call->args[1]);
      PrimExpr next = Cast(expected->dtype, expected + delta);
      return BufferStore(expected->buffer, next, expected->indices);
    }
    if (call->op.same_as(dist_completion_init_())) {
      return InitializeCompletion(call);
    }
    if (call->op.same_as(dist_wait_any_())) {
      Var index("dist_wait_any_index", DataType::Int(32));
      return LetStmt(index, evaluate_node->value,
                     ConsumeWaitAny(call, index, Evaluate(0)));
    }
    return StmtExprMutator::VisitStmt_(evaluate_node);
  }

  Stmt VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
    if (!call || !call->op.same_as(dist_wait_any_())) {
      return StmtExprMutator::VisitStmt_(let_node);
    }
    ICHECK_EQ(call->args.size(), 6U);
    Stmt body = VisitStmt(let_node->body);
    return LetStmt(let_node->var, let_node->value,
                   ConsumeWaitAny(call, let_node->var, body), let_node->span);
  }

  Stmt ConsumeWaitAny(const CallNode *call, const PrimExpr &index, Stmt body) {
    ICHECK_EQ(call->args.size(), 6U);
    BufferRegion pending = NormalizeToBufferRegion(call->args[4]);
    BufferLoad pending_count =
        RequireLocalState(call->args[5], "pending count");
    PrimExpr pending_index = pending->region[0]->min + index;
    return SeqStmt::Flatten(Array<Stmt>{
        BufferStore(pending->buffer, make_zero(pending->buffer->dtype),
                    {pending_index}),
        BufferStore(pending_count->buffer,
                    pending_count - make_const(pending_count->dtype, 1),
                    pending_count->indices),
        body,
    });
  }

  Stmt InitializeCompletion(const CallNode *call) {
    ICHECK_GE(call->args.size(), 6U);
    BufferRegion signal_expect = NormalizeToBufferRegion(call->args[0]);
    BufferRegion completion_expect = NormalizeToBufferRegion(call->args[1]);
    BufferRegion pending = NormalizeToBufferRegion(call->args[2]);
    BufferLoad pending_count =
        RequireLocalState(call->args[3], "pending count");
    PrimExpr advance_expected = call->args[4];
    ICHECK(advance_expected.dtype().is_bool());

    ICHECK_EQ(signal_expect->region.size(), 1U);
    ICHECK_EQ(completion_expect->region.size(), 1U);
    ICHECK_EQ(pending->region.size(), 1U);
    const auto *count_imm = signal_expect->region[0]->extent.as<IntImmNode>();
    ICHECK(count_imm);
    int64_t endpoint_count = count_imm->value;
    ICHECK_EQ(call->args.size(), static_cast<size_t>(5 + endpoint_count));
    const auto *completion_count =
        completion_expect->region[0]->extent.as<IntImmNode>();
    const auto *pending_extent = pending->region[0]->extent.as<IntImmNode>();
    ICHECK(completion_count && pending_extent);
    ICHECK_EQ(completion_count->value, endpoint_count);
    ICHECK_EQ(pending_extent->value, endpoint_count);

    Array<Stmt> statements;
    PrimExpr total = IntImm(DataType::Int(32), 0);
    for (int64_t index = 0; index < endpoint_count; ++index) {
      PrimExpr delta = call->args[5 + index];
      ICHECK(delta.dtype().is_int() || delta.dtype().is_uint());
      PrimExpr active = delta > make_zero(delta.dtype());
      PrimExpr signal_index =
          signal_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr completion_index =
          completion_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr pending_index =
          pending->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr current = BufferLoad(signal_expect->buffer, {signal_index});
      PrimExpr next =
          Cast(signal_expect->buffer->dtype,
               current + Select(advance_expected,
                                Cast(signal_expect->buffer->dtype, delta),
                                make_zero(signal_expect->buffer->dtype)));
      statements.push_back(
          BufferStore(completion_expect->buffer, next, {completion_index}));
      statements.push_back(
          BufferStore(signal_expect->buffer, next, {signal_index}));
      statements.push_back(BufferStore(pending->buffer,
                                       Cast(pending->buffer->dtype, active),
                                       {pending_index}));
      total = total + Cast(DataType::Int(32), active);
    }
    statements.push_back(
        BufferStore(pending_count->buffer, total, pending_count->indices));
    return SeqStmt::Flatten(statements);
  }

  BufferLoad RequireLocalState(const PrimExpr &expr, const char *name) {
    const auto *load = expr.as<BufferLoadNode>();
    ICHECK(load) << "T.dist " << name << " state must be a BufferLoad, got "
                 << expr;
    ICHECK((load->dtype.is_int() || load->dtype.is_uint()) &&
           (load->dtype.bits() == 8 || load->dtype.bits() == 32))
        << "T.dist " << name << " state must use an 8- or 32-bit integer dtype";
    return tvm::ffi::GetRef<BufferLoad>(load);
  }

  std::unordered_set<const EvaluateNode *> empty_submits_;
};

PrimFunc Run(PrimFunc func) {
  if (!IsMultiRank(func)) {
    return func;
  }
  DistOpDetector detector(/*high_level=*/false);
  if (!detector.Detect(func->body)) {
    return func;
  }
  DistQueueValidator::Result queue = DistQueueValidator::Validate(func->body);
  // Inject at the recorded nodes before signal lowering rebuilds the tree.
  if (!queue.returns_needing_wait.empty() ||
      !queue.core_exits_needing_wait.empty()) {
    func = InjectDistExitWait::Run(std::move(func), queue);
  }
  func = InjectDistSignalAdvance::Run(std::move(func),
                                      std::move(queue.empty_submits));
  return func;
}

} // namespace

tvm::transform::Pass InjectDistSync() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectDistSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectDistSync", InjectDistSync);
}

} // namespace tl
} // namespace tvm
