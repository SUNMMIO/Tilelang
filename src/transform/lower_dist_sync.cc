/*!
 * \file tl/transform/lower_dist_sync.cc
 * \brief Materialize Rank submit synchronization and signal-state updates.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

using BufferRegions = std::vector<BufferRegion>;

bool SameBufferRegion(const BufferRegion &lhs, const BufferRegion &rhs) {
  return StructuralEqual()(lhs, rhs);
}

void AddUniqueRegion(BufferRegions *regions, const BufferRegion &region) {
  for (const BufferRegion &existing : *regions) {
    if (SameBufferRegion(existing, region)) {
      return;
    }
  }
  regions->push_back(region);
}

BufferRegions MergeRegions(const BufferRegions &lhs, const BufferRegions &rhs) {
  BufferRegions merged = lhs;
  for (const BufferRegion &region : rhs) {
    AddUniqueRegion(&merged, region);
  }
  return merged;
}

BufferRegions WidenLoopSourceRegions(
    const BufferRegions &regions,
    const std::unordered_set<const VarNode *> &loop_local_vars) {
  BufferRegions widened;
  for (const BufferRegion &region : regions) {
    Region bounds;
    for (size_t dim = 0; dim < region->region.size(); ++dim) {
      const Range &range = region->region[dim];
      bool changes = false;
      auto check = [&](const ObjectRef &node) {
        if (const auto *var = node.as<VarNode>()) {
          changes |= loop_local_vars.count(var) != 0;
        }
        changes |= node.as<BufferLoadNode>() != nullptr;
      };
      PostOrderVisit(range->min, check);
      PostOrderVisit(range->extent, check);
      // An address captured by an earlier send must not be compared using the
      // next iteration's values. Retain invariant dimensions, but
      // conservatively cover a varying dimension until cross-iteration bounds
      // are available.
      bounds.push_back(changes
                           ? Range::FromMinExtent(make_zero(range->min.dtype()),
                                                  region->buffer->shape[dim])
                           : range);
    }
    AddUniqueRegion(&widened, BufferRegion(region->buffer, std::move(bounds)));
  }
  return widened;
}

bool SameRegionSet(const BufferRegions &lhs, const BufferRegions &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (const BufferRegion &region : lhs) {
    bool found = false;
    for (const BufferRegion &candidate : rhs) {
      found |= SameBufferRegion(region, candidate);
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

bool AnyRegionOverlaps(const BufferRegions &pending,
                       const BufferRegions &accesses) {
  for (const BufferRegion &lhs : pending) {
    for (const BufferRegion &rhs : accesses) {
      if (BufferRegionsMayOverlap(lhs->buffer, lhs->region, rhs->buffer,
                                  rhs->region)) {
        return true;
      }
    }
  }
  return false;
}

bool AnyRegionUsesBuffer(const BufferRegions &regions, const Var &buffer_var) {
  for (const BufferRegion &region : regions) {
    if (region->buffer->data.same_as(buffer_var)) {
      return true;
    }
  }
  return false;
}

BufferRegions CallWriteRegions(const CallNode *call) {
  BufferRegions writes;
  auto add = [&](size_t index) {
    ICHECK_LT(index, call->args.size());
    AddUniqueRegion(&writes, NormalizeToBufferRegion(call->args[index]));
  };
  if (call->op.same_as(dma_copy()) ||
      call->op.same_as(sunmmio_layout_transform()) ||
      call->op.same_as(sunmmio_transpose())) {
    add(1);
  } else if (call->op.same_as(mma_sunmmio())) {
    add(2);
  } else if (call->op.same_as(broadcast_())) {
    add(kBroadcastArgDst);
  } else if (call->op.same_as(mx_pack())) {
    add(2);
  } else if (call->op.same_as(mx_unpack())) {
    add(1);
    add(2);
  }
  return writes;
}

class WriteRegionCollector : public StmtExprVisitor {
public:
  BufferRegions Collect(const Stmt &stmt) {
    VisitStmt(stmt);
    return std::move(writes_);
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    Region region;
    for (const PrimExpr &index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    AddUniqueRegion(&writes_, BufferRegion(op->buffer, std::move(region)));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    for (const BufferRegion &region : CallWriteRegions(op)) {
      AddUniqueRegion(&writes_, region);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferRegions writes_;
};

// Track the implicit per-core send queue across structured control flow.
class DistQueueValidator : public StmtVisitor {
public:
  struct Result {
    std::unordered_set<const EvaluateNode *> returns_needing_wait;
    std::unordered_set<const AttrStmtNode *> core_exits_needing_wait;
    std::unordered_set<const EvaluateNode *> evaluates_needing_wait;
    std::unordered_set<const BufferStoreNode *> stores_needing_wait;
    std::unordered_set<const AllocateNode *> allocations_needing_wait;
    std::unordered_set<const ForNode *> for_loops_needing_wait;
    std::unordered_set<const WhileNode *> while_loops_needing_wait;
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
                  std::move(validator.evaluates_needing_wait_),
                  std::move(validator.stores_needing_wait_),
                  std::move(validator.allocations_needing_wait_),
                  std::move(validator.for_loops_needing_wait_),
                  std::move(validator.while_loops_needing_wait_),
                  std::move(validator.empty_submits_)};
  }

private:
  struct State {
    bool queue_pending;
    bool strict_batch_open;
    bool channel_in_flight;
    bool falls_through;
    BufferRegions queued_sources;
    BufferRegions inflight_sources;
  };

  struct LoopExits {
    State breaks{false, false, false, false};
    State continues{false, false, false, false};
  };

  State GetState() const {
    return State{queue_pending_, strict_batch_open_, channel_in_flight_,
                 falls_through_, queued_sources_,    inflight_sources_};
  }

  void SetState(const State &state) {
    queue_pending_ = state.queue_pending;
    strict_batch_open_ = state.strict_batch_open;
    channel_in_flight_ = state.channel_in_flight;
    falls_through_ = state.falls_through;
    queued_sources_ = state.queued_sources;
    inflight_sources_ = state.inflight_sources;
  }

  static State Join(const State &lhs, const State &rhs) {
    if (!lhs.falls_through) {
      return rhs;
    }
    if (!rhs.falls_through) {
      return lhs;
    }
    ICHECK_EQ(lhs.strict_batch_open, rhs.strict_batch_open);
    return State{lhs.queue_pending || rhs.queue_pending,
                 lhs.strict_batch_open,
                 lhs.channel_in_flight || rhs.channel_in_flight,
                 true,
                 MergeRegions(lhs.queued_sources, rhs.queued_sources),
                 MergeRegions(lhs.inflight_sources, rhs.inflight_sources)};
  }

  static bool SameLoopState(const State &lhs, const State &rhs) {
    return lhs.queue_pending == rhs.queue_pending &&
           lhs.channel_in_flight == rhs.channel_in_flight &&
           SameRegionSet(lhs.queued_sources, rhs.queued_sources) &&
           SameRegionSet(lhs.inflight_sources, rhs.inflight_sources);
  }

  void CheckSubmitted(bool before_return) const {
    ICHECK(!strict_batch_open_)
        << "T.dist submit=True batch is missing its submit boundary"
        << (before_return ? " before return" : "");
    ICHECK(!queue_pending_) << "T.dist send queue is not submitted"
                            << (before_return ? " before return" : "")
                            << "; call T.dist.submit() or use submit=True";
  }

  bool HandleWrites(const BufferRegions &writes, const char *description) {
    if (writes.empty()) {
      return false;
    }
    ICHECK(!AnyRegionOverlaps(queued_sources_, writes))
        << "T.dist source is modified before its queued batch is submitted ("
        << description << ")";
    if (!AnyRegionOverlaps(inflight_sources_, writes)) {
      return false;
    }
    // dist_wait_send completes the whole local send channel, not just the
    // conflicting region.
    inflight_sources_.clear();
    channel_in_flight_ = false;
    return true;
  }

  bool UsesAllocationLeavingCurrentLoop(const BufferRegions &regions) const {
    ICHECK(!loop_allocation_depths_.empty());
    size_t first_exiting_allocation = loop_allocation_depths_.back();
    for (size_t i = first_exiting_allocation; i < active_allocations_.size();
         ++i) {
      if (AnyRegionUsesBuffer(regions, active_allocations_[i])) {
        return true;
      }
    }
    return false;
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
        inflight_sources_.clear();
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

  void VisitLoop(const Stmt &body, const Var &loop_var = Var()) {
    std::unordered_set<const VarNode *> loop_local_vars;
    if (loop_var.defined()) {
      loop_local_vars.insert(loop_var.get());
    }
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *loop = node.as<ForNode>()) {
        loop_local_vars.insert(loop->loop_var.get());
      } else if (const auto *let = node.as<LetStmtNode>()) {
        loop_local_vars.insert(let->var.get());
      } else if (const auto *let = node.as<LetNode>()) {
        loop_local_vars.insert(let->var.get());
      }
    });
    auto widen = [&](State state) {
      state.queued_sources =
          WidenLoopSourceRegions(state.queued_sources, loop_local_vars);
      state.inflight_sources =
          WidenLoopSourceRegions(state.inflight_sources, loop_local_vars);
      return state;
    };
    State incoming = GetState();
    State entry = incoming;
    loop_exits_.emplace_back();
    loop_allocation_depths_.push_back(active_allocations_.size());
    while (true) {
      loop_exits_.back() = LoopExits{};
      State body_state = VisitBranch(body, entry);
      // Continue reaches the next iteration; break only reaches the loop exit.
      State backedge = Join(body_state, loop_exits_.back().continues);
      State next = Join(incoming, widen(std::move(backedge)));
      if (SameLoopState(next, entry)) {
        // Include zero iterations and exhaustion after a normal/continue path.
        SetState(Join(next, widen(loop_exits_.back().breaks)));
        break;
      }
      entry = next;
    }
    loop_allocation_depths_.pop_back();
    loop_exits_.pop_back();
  }

  void VisitStmt_(const ForNode *op) final {
    BufferRegions writes = WriteRegionCollector().Collect(op->body);
    if (AnyRegionOverlaps(inflight_sources_, writes)) {
      for_loops_needing_wait_.insert(op);
      inflight_sources_.clear();
      channel_in_flight_ = false;
    }
    VisitExpr(op->min);
    VisitExpr(op->extent);
    VisitLoop(op->body, op->loop_var);
  }

  void VisitStmt_(const WhileNode *op) final {
    BufferRegions writes = WriteRegionCollector().Collect(op->body);
    if (AnyRegionOverlaps(inflight_sources_, writes)) {
      while_loops_needing_wait_.insert(op);
      inflight_sources_.clear();
      channel_in_flight_ = false;
    }
    VisitExpr(op->condition);
    VisitLoop(op->body);
  }

  void VisitStmt_(const AllocateNode *op) final {
    for (const PrimExpr &extent : op->extents) {
      VisitExpr(extent);
    }
    VisitExpr(op->condition);
    active_allocations_.push_back(op->buffer_var);
    VisitStmt(op->body);
    active_allocations_.pop_back();
    if (!falls_through_) {
      return;
    }
    if (AnyRegionUsesBuffer(queued_sources_, op->buffer_var)) {
      CheckSubmitted(/*before_return=*/false);
    }
    if (AnyRegionUsesBuffer(inflight_sources_, op->buffer_var)) {
      allocations_needing_wait_.insert(op);
      inflight_sources_.clear();
      channel_in_flight_ = false;
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Region region;
    for (const PrimExpr &index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    BufferRegions writes{BufferRegion(op->buffer, std::move(region))};
    if (HandleWrites(writes, "buffer store")) {
      stores_needing_wait_.insert(op);
    }
    StmtVisitor::VisitStmt_(op);
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
    if (call->op.same_as(builtin::break_loop()) ||
        call->op.same_as(builtin::continue_loop())) {
      bool is_break = call->op.same_as(builtin::break_loop());
      ICHECK(!loop_exits_.empty())
          << "T.dist queue analysis: loop jump outside a loop";
      ICHECK(!strict_batch_open_)
          << "T.dist submit=True batch is missing its submit boundary before "
          << (is_break ? "break" : "continue");
      ICHECK(!UsesAllocationLeavingCurrentLoop(queued_sources_))
          << "T.dist source is released by "
          << (is_break ? "break" : "continue")
          << " before its queued batch is submitted";
      if (UsesAllocationLeavingCurrentLoop(inflight_sources_)) {
        evaluates_needing_wait_.insert(op);
        inflight_sources_.clear();
        channel_in_flight_ = false;
      }
      State &exit =
          is_break ? loop_exits_.back().breaks : loop_exits_.back().continues;
      exit = Join(exit, GetState());
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
    if (call->op.same_as(dist_put_())) {
      ICHECK_EQ(call->args.size(), 6U);
      AddUniqueRegion(&queued_sources_, NormalizeToBufferRegion(call->args[0]));
      queue_pending_ = true;
      return;
    }
    if (call->op.same_as(dist_signal_put_())) {
      queue_pending_ = true;
      return;
    }
    if (call->op.same_as(dist_submit_())) {
      bool nonempty = queue_pending_;
      if (!nonempty && loop_exits_.empty()) {
        empty_submits_.insert(op);
        // Removing an empty submit also removes its channel-idle wait. Keep
        // earlier in-flight sources protected until a real wait completes them.
        strict_batch_open_ = false;
        return;
      }
      // Every retained submit receives a channel-idle wait during rewriting,
      // so prior in-flight sources are complete before this batch starts.
      inflight_sources_.clear();
      if (nonempty) {
        inflight_sources_ = queued_sources_;
      }
      queued_sources_.clear();
      queue_pending_ = false;
      strict_batch_open_ = false;
      channel_in_flight_ |= nonempty;
      return;
    }
    if (call->op.same_as(dist_wait_send())) {
      inflight_sources_.clear();
      channel_in_flight_ = false;
      return;
    }
    if (HandleWrites(CallWriteRegions(call), "asynchronous destination")) {
      evaluates_needing_wait_.insert(op);
    }
    StmtVisitor::VisitStmt_(op);
  }

  bool queue_pending_{false};
  bool strict_batch_open_{false};
  bool channel_in_flight_{false};
  bool falls_through_{true};
  BufferRegions queued_sources_;
  BufferRegions inflight_sources_;
  std::vector<LoopExits> loop_exits_;
  std::vector<size_t> loop_allocation_depths_;
  std::vector<Var> active_allocations_;
  int core_depth_{0};
  std::unordered_set<const EvaluateNode *> returns_needing_wait_;
  std::unordered_set<const AttrStmtNode *> core_exits_needing_wait_;
  std::unordered_set<const EvaluateNode *> evaluates_needing_wait_;
  std::unordered_set<const BufferStoreNode *> stores_needing_wait_;
  std::unordered_set<const AllocateNode *> allocations_needing_wait_;
  std::unordered_set<const ForNode *> for_loops_needing_wait_;
  std::unordered_set<const WhileNode *> while_loops_needing_wait_;
  std::unordered_set<const EvaluateNode *> empty_submits_;
};

// Materialize the sender waits selected by the queue and source-lifetime
// analysis before signal-state lowering rebuilds the tree.
class InjectDistSenderWait : public StmtExprMutator {
public:
  static PrimFunc Run(PrimFunc func, const DistQueueValidator::Result &queue) {
    InjectDistSenderWait injector(queue);
    Stmt body = injector(func->body);
    func.CopyOnWrite()->body = std::move(body);
    return func;
  }

private:
  explicit InjectDistSenderWait(const DistQueueValidator::Result &queue)
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

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    Array<PrimExpr> iter_values;
    iter_values.reserve(op->iter_values.size());
    for (const PrimExpr &value : op->iter_values) {
      iter_values.push_back(VisitExpr(value));
    }
    PrimExpr predicate = VisitExpr(op->predicate);
    Block block = op->block;
    BlockNode *writer = block.CopyOnWrite();
    if (writer->init.defined()) {
      writer->init = VisitStmt(writer->init.value());
    }
    writer->body = VisitStmt(writer->body);
    return BlockRealize(iter_values, predicate, block, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    Stmt body = VisitStmt(op->body);
    if (queue_.core_exits_needing_wait.count(op)) {
      body = AppendCoreWait(body);
    }
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    Stmt body = VisitStmt(op->body);
    if (queue_.allocations_needing_wait.count(op)) {
      body = AppendCoreWait(body);
    }
    return Allocate(op->buffer_var, op->dtype, op->extents, op->condition, body,
                    op->annotations, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt store = StmtExprMutator::VisitStmt_(op);
    if (!queue_.stores_needing_wait.count(op)) {
      return store;
    }
    return SeqStmt::Flatten(Array<Stmt>{WaitSend(), std::move(store)});
  }

  Stmt VisitStmt_(const ForNode *op) final {
    Stmt loop = For(op->loop_var, VisitExpr(op->min), VisitExpr(op->extent),
                    op->kind, VisitStmt(op->body), op->thread_binding,
                    op->annotations, op->step, op->span);
    if (!queue_.for_loops_needing_wait.count(op)) {
      return loop;
    }
    return SeqStmt::Flatten(Array<Stmt>{WaitSend(), std::move(loop)});
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    Stmt loop = While(VisitExpr(op->condition), VisitStmt(op->body), op->span);
    if (!queue_.while_loops_needing_wait.count(op)) {
      return loop;
    }
    return SeqStmt::Flatten(Array<Stmt>{WaitSend(), std::move(loop)});
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt evaluate = StmtExprMutator::VisitStmt_(op);
    if (queue_.returns_needing_wait.count(op) ||
        queue_.evaluates_needing_wait.count(op)) {
      return SeqStmt::Flatten(Array<Stmt>{WaitSend(), std::move(evaluate)});
    }
    return evaluate;
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
  PrimExpr CastSignalDelta(DataType dtype, PrimExpr delta) {
    if (dtype == DataType::UInt(8) && delta.dtype().bits() > 8) {
      // TVM's constant cast rejects uint8(256). Normalize before narrowing,
      // including when a dynamic delta later becomes constant during lowering.
      if (const auto *value = delta.as<IntImmNode>()) {
        return make_const(dtype, value->value & 255);
      }
      delta = bitwise_and(delta, make_const(delta.dtype(), 255));
    }
    return Cast(dtype, delta);
  }

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
             "LowerDistSync";
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
      // Convert to the signal width before arithmetic: uint8 wraps modulo 256,
      // while uint64 memory expectations retain accumulated high bits.
      PrimExpr delta = CastSignalDelta(expected->dtype, call->args[1]);
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
      // A positive delta may wrap to zero at the signal width (e.g. 256).
      // Pending membership uses the original delta, not the narrowed value.
      PrimExpr active = delta > make_zero(delta.dtype());
      PrimExpr signal_index =
          signal_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr completion_index =
          completion_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr pending_index =
          pending->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr current = BufferLoad(signal_expect->buffer, {signal_index});
      PrimExpr typed_delta =
          CastSignalDelta(signal_expect->buffer->dtype, delta);
      PrimExpr next =
          Cast(signal_expect->buffer->dtype,
               current + Select(advance_expected, typed_delta,
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
           (load->dtype.bits() == 8 || load->dtype.bits() == 32 ||
            load->dtype.bits() == 64))
        << "T.dist " << name
        << " state must use an 8-, 32-, or 64-bit integer dtype";
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
      !queue.core_exits_needing_wait.empty() ||
      !queue.evaluates_needing_wait.empty() ||
      !queue.stores_needing_wait.empty() ||
      !queue.allocations_needing_wait.empty() ||
      !queue.for_loops_needing_wait.empty() ||
      !queue.while_loops_needing_wait.empty()) {
    func = InjectDistSenderWait::Run(std::move(func), queue);
  }
  func = InjectDistSignalAdvance::Run(std::move(func),
                                      std::move(queue.empty_submits));
  return func;
}

} // namespace

tvm::transform::Pass LowerDistSync() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerDistSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerDistSync", LowerDistSync);
}

} // namespace tl
} // namespace tvm
