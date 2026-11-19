/*!
 * \file tl/transform/lower_dist_sync.cc
 * \brief Materialize Rank submit synchronization and signal-state updates.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/utils.h"
#include "../support/finite_width_integer.h"
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
  WriteRegionCollector(const Map<Buffer, Layout> &layouts,
                       const SunmmioTileProcessorConfig &config)
      : layouts_(layouts), config_(config) {}
  BufferRegions Collect(const Stmt &stmt) {
    VisitStmt(stmt);
    return std::move(writes_);
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    Region region = GetSunmmioScalarTileAccessRegion(op->buffer, op->indices,
                                                     layouts_, config_);
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
  const Map<Buffer, Layout> &layouts_;
  const SunmmioTileProcessorConfig &config_;
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

  static Result Validate(const PrimFunc &func) {
    DistQueueValidator validator(func);
    validator(func->body);
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
  explicit DistQueueValidator(const PrimFunc &func)
      : layouts_(func->GetAttr<Map<Buffer, Layout>>(attr::kLayoutMap)
                     .value_or(Map<Buffer, Layout>())),
        config_(GetSunmmioTileProcessorConfig(
            func->GetAttr<Target>(tvm::attr::kTarget))) {}

  Map<Buffer, Layout> layouts_;
  SunmmioTileProcessorConfig config_;
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
    bool needs_wait = AnyRegionOverlaps(inflight_sources_, writes);
    if (needs_wait) {
      // dist_wait_send completes the whole local send channel, not just the
      // conflicting region.
      inflight_sources_.clear();
      channel_in_flight_ = false;
    }
    for (BufferRegions *sources : {&queued_sources_, &inflight_sources_}) {
      for (BufferRegion &source : *sources) {
        source = BufferRegion(
            source->buffer,
            WidenRegionForIndexWrites(source->buffer, source->region, writes));
      }
    }
    return needs_wait;
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
    BufferRegions writes =
        WriteRegionCollector(layouts_, config_).Collect(op->body);
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
    BufferRegions writes =
        WriteRegionCollector(layouts_, config_).Collect(op->body);
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
    Region region = GetSunmmioScalarTileAccessRegion(op->buffer, op->indices,
                                                     layouts_, config_);
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

// The emitting visitor registers the original growth sites. Only their integer
// dependencies are summarized; payload code is never symbolically executed.
// State contains numeric intervals/snapshots, not accumulated TIR expressions.
class ExpectedFinalBounds {
public:
  explicit ExpectedFinalBounds(const PrimFunc &func)
      : source_(func->body),
        enabled_(func->attrs.defined() &&
                 func->attrs->dict.count(kDistSignalCountsAttr)) {
    if (auto index = func->GetAttr<Integer>("tl.dist.rank_id_param_index")) {
      auto world = func->GetAttr<Integer>(kDistWorldSizeAttr);
      if (world)
        analyzer_.Bind(func->params[index.value()->value],
                       Range::FromMinExtent(0, world.value()));
    }
  }

  void Register(const CallNode *call, const BufferLoad &target,
                const PrimExpr &delta) {
    if (!enabled_)
      return; // Isolated internal materialization tests have no signal plan.
    auto annotation = call->annotations.Get(kDistExpectedSignalKindAttr);
    ICHECK(annotation) << "Missing expected signal kind in LowerDistSync";
    const auto &kind = RequireDistSignalKindInfo(
        Downcast<PrimExpr>(annotation.value()), "expected signal kind");
    bool flag = kind.update_mode != DistSignalUpdateMode::kMemory;
    if (is_zero(delta))
      return;
    if (!flag &&
        (delta.dtype().is_uint() ||
         (SideEffect(delta) == CallEffectKind::kPure && Eval(delta).lo >= 0)))
      return; // No MEMORY cumulative bound; unsigned deltas are non-negative.
    updates_[call].push_back({target, delta, flag});
    Collect(delta);
    if (flag) {
      Add(target->buffer);
      counters_.insert(target->buffer->data.get());
      for (const PrimExpr &index : target->indices)
        Collect(index);
    }
  }

  void Check() {
    if (updates_.empty())
      return;
    // Build a dependency slice once. Conditions are included only when they
    // control a relevant write/update, unlike a scan of all integer payload.
    PostOrderVisit(source_, [&](const ObjectRef &node) {
      if (const auto *let = node.as<LetStmtNode>())
        lets_.emplace(let->var.get(), let->value);
      if (const auto *call = node.as<CallNode>();
          call && call->op.same_as(dist_put_()))
        remote_.insert(
            NormalizeToBufferRegion(call->args[1])->buffer->data.get());
    });
    size_t old_size;
    do {
      old_size = memory_.size() + needed_vars_.size();
      for (const auto &[var, expr] : lets_)
        if (needed_vars_.count(var))
          Collect(expr);
      PostOrderVisit(source_, [&](const ObjectRef &node) {
        if (const auto *store = node.as<BufferStoreNode>();
            store && memory_.count(store->buffer->data.get())) {
          Collect(store->value);
          for (const PrimExpr &index : store->indices)
            Collect(index);
        }
      });
      relevant_.clear();
      Mark(source_);
    } while (old_size != memory_.size() + needed_vars_.size());
    // Build once; loops apply this small effect summary using numeric facts.
    // The ordinary emitter has already produced the runtime code once.
    Build(source_)();
    ICHECK(nonnegative_)
        << "T.dist cannot prove expected delta is non-negative";
    for (const auto &[data, upper] : peaks_)
      ICHECK(upper <= 255)
          << "T.dist cannot prove cumulative expected is in [0, 255] for "
          << memory_.at(data).buffer->name << ": final upper bound " << upper;
  }

private:
  static constexpr int64_t kInf = arith::ConstIntBound::kPosInf;
  static constexpr int64_t kNegInf = arith::ConstIntBound::kNegInf;
  static constexpr size_t kWorkLimit = 1 << 20;
  struct Bounds {
    int64_t lo, hi;
  };
  struct Cell {
    Bounds range;
    PrimExpr snapshot;
  };
  struct Memory {
    Buffer buffer;
    std::vector<Cell> cells;
    std::vector<std::pair<PrimExpr, PrimExpr>> reads;
  };
  struct Update {
    BufferLoad target;
    PrimExpr delta;
    bool flag;
  };
  using State = std::unordered_map<const VarNode *, Memory>;
  using Effect = std::function<void()>;

  void Work(size_t count = 1) {
    work_ += count;
    ICHECK(work_ <= kWorkLimit) << "T.dist cannot prove cumulative expected "
                                   "within final-bound analysis budget";
  }
  Cell Snapshot(Bounds range, DataType dtype) {
    if (range.lo == range.hi && range.lo != kInf && range.lo != kNegInf)
      return {range, make_const(dtype, range.lo)};
    Var value("dist_bound", dtype);
    analyzer_.const_int_bound.Update(value,
                                     arith::ConstIntBound(range.lo, range.hi));
    return {range, value};
  }
  Cell Unknown(DataType dtype) {
    Var value("dist_unknown", dtype);
    auto b = analyzer_.const_int_bound(value);
    return {{b->min_value, b->max_value}, value};
  }
  void Add(const Buffer &buffer) {
    if (memory_.count(buffer->data.get()) ||
        !(buffer->dtype.is_int() || buffer->dtype.is_uint()))
      return;
    int64_t count = 1;
    for (const PrimExpr &dim : buffer->shape) {
      const auto *value = dim.as<IntImmNode>();
      if (!value || value->value <= 0 || value->value > 4096 / count)
        return;
      count *= value->value;
    }
    Memory entry{buffer, {}, {}};
    for (int64_t i = 0; i < count; ++i)
      entry.cells.push_back(Unknown(buffer->dtype));
    memory_.emplace(buffer->data.get(), std::move(entry));
  }
  void Collect(const PrimExpr &expr) {
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>())
        Add(load->buffer);
      if (const auto *var = node.as<VarNode>())
        needed_vars_.insert(var);
    });
  }
  std::vector<const VarNode *> LocalWrites(const CallNode *call) {
    std::vector<const VarNode *> result;
    auto add = [&](const VarNode *data) {
      if (memory_.count(data) && !counters_.count(data))
        result.push_back(data);
    };
    if (call->op.same_as(dist_completion_init_())) {
      add(NormalizeToBufferRegion(call->args[1])->buffer->data.get());
      add(NormalizeToBufferRegion(call->args[2])->buffer->data.get());
      add(Downcast<BufferLoad>(call->args[3])->buffer->data.get());
    } else if (call->op.same_as(dist_wait_any_())) {
      add(NormalizeToBufferRegion(call->args[4])->buffer->data.get());
      add(Downcast<BufferLoad>(call->args[5])->buffer->data.get());
    } else if (!IsLeafDistOp(call)) {
      for (const PrimExpr &arg : call->args) {
        if (const auto *region = arg.as<CallNode>();
            region && region->op.same_as(RegionOp::Get())) {
          RegionOp view(region->args);
          if (view->GetAccessMask() & 2)
            add(view->GetBuffer()->data.get());
        } else if (const auto *var = arg.as<VarNode>();
                   var && var->dtype.is_handle() &&
                   SideEffect(ffi::GetRef<PrimExpr>(call)) >
                       CallEffectKind::kReadState)
          add(var);
      }
    }
    return result;
  }
  bool Writes(const CallNode *call) { return !LocalWrites(call).empty(); }
  void Invalidate(const CallNode *call) {
    for (const auto *data : LocalWrites(call))
      Reset(data);
    // Public Completion progress maintains a non-negative logical count.
    if (call->op.same_as(dist_completion_init_())) {
      auto buffer = Downcast<BufferLoad>(call->args[3])->buffer;
      auto count = NormalizeToBufferRegion(call->args[0])->region[0]->extent;
      Reset(buffer->data.get(),
            Bounds{0, RequireIntImm(count, "completion count")});
    } else if (call->op.same_as(dist_wait_any_())) {
      auto buffer = Downcast<BufferLoad>(call->args[5])->buffer;
      Reset(buffer->data.get(),
            Bounds{0, RequireIntImm(call->args[2], "completion count")});
    }
  }
  bool Mark(const Stmt &stmt) {
    bool found = false;
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq)
        found |= Mark(s);
    } else if (const auto *op = stmt.as<IfThenElseNode>()) {
      found = Mark(op->then_case);
      if (op->else_case)
        found |= Mark(op->else_case.value());
      if (found)
        Collect(op->condition);
    } else if (const auto *op = stmt.as<ForNode>()) {
      found = Mark(op->body);
      if (found) {
        Collect(op->min);
        Collect(op->extent);
      }
    } else if (const auto *op = stmt.as<WhileNode>()) {
      found = Mark(op->body);
      if (found)
        Collect(op->condition);
    } else if (const auto *op = stmt.as<LetStmtNode>()) {
      found = Mark(op->body) || needed_vars_.count(op->var.get());
      if (needed_vars_.count(op->var.get()))
        Collect(op->value);
    } else if (const auto *op = stmt.as<AttrStmtNode>())
      found = Mark(op->body);
    else if (const auto *op = stmt.as<AllocateNode>())
      found = Mark(op->body);
    else if (const auto *op = stmt.as<DeclBufferNode>())
      found = Mark(op->body);
    else if (const auto *op = stmt.as<AssertStmtNode>())
      found = Mark(op->body);
    else if (const auto *op = stmt.as<BlockRealizeNode>())
      found = Mark(op->block);
    else if (const auto *op = stmt.as<BlockNode>()) {
      found = Mark(op->body);
      if (op->init)
        found |= Mark(op->init.value());
    } else if (const auto *op = stmt.as<BufferStoreNode>())
      found = memory_.count(op->buffer->data.get());
    else if (const auto *op = stmt.as<EvaluateNode>()) {
      if (const auto *call = op->value.as<CallNode>())
        found = updates_.count(call) || Writes(call);
    }
    if (found)
      relevant_.insert(stmt.get());
    return found;
  }
  class Resolver : public ExprMutator {
  public:
    explicit Resolver(ExpectedFinalBounds *owner, bool condition = false)
        : owner_(owner), integers_(&owner->analyzer_), condition_(condition) {}
    PrimExpr VisitExpr(const PrimExpr &expr) final {
      owner_->Work();
      PrimExpr value = ExprMutator::VisitExpr(expr);
      if (!(value.dtype().is_int() || value.dtype().is_uint() ||
            value.dtype().is_bool()))
        return value;
      if (!condition_)
        return integers_.Normalize(value);
      auto model = integers_.Analyze(value);
      // An unmodeled condition leaves both branches reachable. Delta and loop
      // bound resolution remain strict; only branch facts are optional.
      return model.valid ? model.expr : Var("dist_condition", value.dtype());
    }
    PrimExpr VisitExpr_(const VarNode *op) final {
      auto it = owner_->bindings_.find(op);
      return it == owner_->bindings_.end() ? ffi::GetRef<PrimExpr>(op)
                                           : it->second;
    }
    PrimExpr VisitExpr_(const BufferLoadNode *op) final {
      Array<PrimExpr> indices;
      for (const PrimExpr &index : op->indices)
        indices.push_back(VisitExpr(index));
      return owner_->Read(op->buffer, indices);
    }
    PrimExpr VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(builtin::if_then_else())) {
        PrimExpr condition = owner_->analyzer_.Simplify(VisitExpr(op->args[0]));
        if (is_one(condition))
          return VisitExpr(op->args[1]);
        if (is_zero(condition))
          return VisitExpr(op->args[2]);
        return Select(condition, VisitExpr(op->args[1]),
                      VisitExpr(op->args[2]));
      }
      return ExprMutator::VisitExpr_(op);
    }

  private:
    ExpectedFinalBounds *owner_;
    FiniteWidthInteger integers_;
    bool condition_;
  };
  PrimExpr Resolve(const PrimExpr &expr) {
    Work();
    return Resolver(this)(expr);
  }
  Bounds Eval(const PrimExpr &expr) {
    if (expr.dtype().is_uint() && expr.dtype().bits() == 64) {
      if (const auto *imm = expr.as<IntImmNode>(); imm && imm->value >= 0)
        return {imm->value, imm->value};
      if (const auto *call = expr.as<CallNode>();
          call && call->op.same_as(builtin::large_uint_imm()))
        return {0, kInf};
    }
    auto bound = analyzer_.const_int_bound(Resolve(expr));
    return {bound->min_value, bound->max_value};
  }
  PrimExpr Index(const Buffer &buffer, const Array<PrimExpr> &indices) {
    if (!buffer->strides.empty() || !is_zero(buffer->elem_offset))
      return Var("dist_index", DataType::Int(64));
    PrimExpr value = make_zero(DataType::Int(64));
    for (size_t i = 0; i < indices.size(); ++i)
      value = value * Cast(DataType::Int(64), buffer->shape[i]) +
              Cast(DataType::Int(64), indices[i]);
    return analyzer_.Simplify(Resolve(value));
  }
  bool SameView(const Memory &entry, const Buffer &buffer) {
    return entry.buffer->dtype == buffer->dtype &&
           StructuralEqual()(entry.buffer->shape, buffer->shape) &&
           StructuralEqual()(entry.buffer->strides, buffer->strides) &&
           StructuralEqual()(entry.buffer->elem_offset, buffer->elem_offset);
  }
  PrimExpr Read(const Buffer &buffer, const Array<PrimExpr> &indices) {
    auto found = memory_.find(buffer->data.get());
    if (found == memory_.end() || remote_.count(buffer->data.get()) ||
        !SameView(found->second, buffer))
      return Unknown(buffer->dtype).snapshot;
    auto &entry = found->second;
    PrimExpr index = Index(buffer, indices);
    if (const auto *i = index.as<IntImmNode>()) {
      if (i->value >= 0 && static_cast<size_t>(i->value) < entry.cells.size())
        return entry.cells[i->value].snapshot;
      return Unknown(buffer->dtype).snapshot;
    }
    for (const auto &[saved, value] : entry.reads)
      if (StructuralEqual()(saved, index))
        return value;
    Bounds range{kInf, kNegInf};
    for (const Cell &cell : entry.cells) {
      range.lo = std::min(range.lo, cell.range.lo);
      range.hi = std::max(range.hi, cell.range.hi);
    }
    PrimExpr value = Snapshot(range, buffer->dtype).snapshot;
    entry.reads.emplace_back(index, value);
    return value;
  }
  void Reset(const VarNode *data, const std::optional<Bounds> &range = {}) {
    auto found = memory_.find(data);
    if (found == memory_.end())
      return;
    auto &entry = found->second;
    entry.reads.clear();
    for (Cell &cell : entry.cells)
      cell = range ? Snapshot(*range, entry.buffer->dtype)
                   : Unknown(entry.buffer->dtype);
  }
  void Write(const Buffer &buffer, const Array<PrimExpr> &indices,
             Bounds value) {
    auto found = memory_.find(buffer->data.get());
    if (found == memory_.end())
      return;
    if (!SameView(found->second, buffer)) {
      Reset(buffer->data.get());
      return;
    }
    auto &entry = found->second;
    entry.reads.clear();
    PrimExpr index = Index(buffer, indices);
    if (const auto *i = index.as<IntImmNode>()) {
      if (i->value >= 0 && static_cast<size_t>(i->value) < entry.cells.size())
        entry.cells[i->value] = Snapshot(value, buffer->dtype);
    } else {
      Work(entry.cells.size());
      for (Cell &cell : entry.cells)
        cell = Snapshot({std::min(cell.range.lo, value.lo),
                         std::max(cell.range.hi, value.hi)},
                        buffer->dtype);
    }
  }
  void Grow(const Update &update) {
    Bounds delta = Eval(update.delta);
    nonnegative_ &= delta.lo >= 0;
    if (!update.flag)
      return;
    auto &entry = memory_.at(update.target->buffer->data.get());
    PrimExpr index = Index(update.target->buffer, update.target->indices);
    auto grow = [&](Cell &cell) {
      int64_t upper = cell.range.hi;
      upper = upper > 255 || delta.hi > 255 || delta.hi < 0
                  ? 256
                  : std::min<int64_t>(256, upper + delta.hi);
      // Only mathematical counter bounds are saturated; runtime storage/casts
      // and auxiliary integer values are unchanged.
      cell = Snapshot({0, upper}, entry.buffer->dtype);
      auto data = update.target->buffer->data.get();
      peaks_[data] = std::max(peaks_[data], upper);
    };
    entry.reads.clear();
    if (const auto *i = index.as<IntImmNode>()) {
      if (i->value >= 0 && static_cast<size_t>(i->value) < entry.cells.size())
        grow(entry.cells[i->value]);
    } else
      for (Cell &cell : entry.cells)
        grow(cell);
  }
  void Join(const State &other) {
    for (auto &[data, entry] : memory_) {
      entry.reads.clear();
      const auto &cells = other.at(data).cells;
      Work(cells.size());
      for (size_t i = 0; i < cells.size(); ++i) {
        if (entry.cells[i].snapshot.same_as(cells[i].snapshot) &&
            entry.cells[i].range.hi == cells[i].range.hi)
          continue;
        Bounds range{std::min(entry.cells[i].range.lo, cells[i].range.lo),
                     std::max(entry.cells[i].range.hi, cells[i].range.hi)};
        entry.cells[i] = Snapshot(range, entry.buffer->dtype);
      }
    }
  }
  void Branch(const PrimExpr &expr, const Effect &yes, const Effect &no) {
    PrimExpr condition = analyzer_.Simplify(Resolver(this, true)(expr));
    if (is_one(condition)) {
      yes();
      return;
    }
    if (is_zero(condition)) {
      no();
      return;
    }
    State incoming = memory_;
    {
      With<arith::ConstraintContext> constraint(&analyzer_, condition);
      yes();
    }
    State left = std::move(memory_);
    memory_ = std::move(incoming);
    {
      With<arith::ConstraintContext> constraint(&analyzer_, Not(condition));
      no();
    }
    Join(left);
  }
  bool InvariantGrowth(const ForNode *loop) {
    bool invariant = true;
    auto inspect = [&](const PrimExpr &expr) {
      PostOrderVisit(expr, [&](const ObjectRef &node) {
        if (node.get() == loop->loop_var.get())
          invariant = false;
        if (const auto *load = node.as<BufferLoadNode>();
            load && counters_.count(load->buffer->data.get()))
          invariant = false;
      });
    };
    PostOrderVisit(loop->body, [&](const ObjectRef &node) {
      if (const auto *store = node.as<BufferStoreNode>();
          store && memory_.count(store->buffer->data.get()))
        invariant = false;
      else if (const auto *call = node.as<CallNode>()) {
        if (Writes(call))
          invariant = false;
        auto it = updates_.find(call);
        if (it != updates_.end())
          for (const Update &u : it->second) {
            inspect(u.delta);
            for (const PrimExpr &index : u.target->indices)
              inspect(index);
          }
      } else if (const auto *branch = node.as<IfThenElseNode>();
                 branch && relevant_.count(branch))
        inspect(branch->condition);
      else if (const auto *nested = node.as<ForNode>();
               nested && relevant_.count(nested)) {
        inspect(nested->min);
        inspect(nested->extent);
        if (nested->step)
          inspect(nested->step.value());
      } else if (const auto *nested = node.as<WhileNode>();
                 nested && relevant_.count(nested))
        inspect(nested->condition);
      else if (const auto *let = node.as<LetStmtNode>();
               let && needed_vars_.count(let->var.get()))
        inspect(let->value);
    });
    return invariant;
  }
  Effect Build(const Stmt &stmt) {
    if (!relevant_.count(stmt.get()))
      return [] {};
    if (const auto *op = stmt.as<SeqStmtNode>()) {
      std::vector<Effect> effects;
      for (const Stmt &s : op->seq)
        if (relevant_.count(s.get()))
          effects.push_back(Build(s));
      return [this, effects] {
        for (const auto &f : effects) {
          Work();
          f();
        }
      };
    }
    if (const auto *op = stmt.as<IfThenElseNode>()) {
      auto yes = Build(op->then_case),
           no = op->else_case ? Build(op->else_case.value()) : Effect([] {});
      return [this, op, yes, no] {
        Work();
        Branch(op->condition, yes, no);
      };
    }
    if (const auto *op = stmt.as<ForNode>()) {
      auto body = Build(op->body);
      return [this, op, body] {
        // scf.for evaluates its bounds once on entry. Keep the resolved
        // expression and its constraints when the body changes that storage.
        PrimExpr entry_min = Resolve(op->min);
        PrimExpr entry_extent = Resolve(op->extent);
        PrimExpr entry_step =
            Resolve(op->step.value_or(make_const(op->min.dtype(), 1)));
        Bounds min = Eval(entry_min), extent = Eval(entry_extent);
        Bounds step = Eval(entry_step);
        ICHECK(step.lo == step.hi && step.lo > 0 && extent.hi != kInf)
            << "T.dist cannot prove cumulative expected for loop bounds";
        int64_t count = extent.hi > 0 ? (extent.hi - 1) / step.lo + 1 : 0;
        if (count > 4096 && InvariantGrowth(op)) {
          State incoming = memory_;
          auto peaks = peaks_;
          for (const auto *data : counters_)
            for (Cell &cell : memory_.at(data).cells)
              cell = Snapshot({0, 0}, memory_.at(data).buffer->dtype);
          body();
          State increments = std::move(memory_);
          memory_ = std::move(incoming);
          peaks_ = std::move(peaks);
          for (const auto *data : counters_) {
            auto &cells = memory_.at(data).cells;
            for (size_t i = 0; i < cells.size(); ++i) {
              auto amount = increments.at(data).cells[i].range.hi;
              auto previous = cells[i].range.hi;
              int64_t upper =
                  previous > 255 || amount > 255 ||
                          (amount > 0 && count > (255 - previous) / amount)
                      ? 256
                      : previous + count * amount;
              cells[i] = Snapshot({0, upper}, memory_.at(data).buffer->dtype);
              peaks_[data] = std::max(peaks_[data], upper);
            }
          }
          return;
        }
        ICHECK(count <= 4096)
            << "T.dist cannot prove cumulative expected for loop extent "
            << op->extent;
        auto saved = bindings_;
        for (int64_t i = 0; i < count; ++i) {
          Work();
          bindings_[op->loop_var.get()] =
              Snapshot({min.lo == kNegInf ? kNegInf : min.lo + i * step.lo,
                        min.hi == kInf ? kInf : min.hi + i * step.lo},
                       op->loop_var.dtype())
                  .snapshot;
          Branch(make_const(entry_extent.dtype(), i * step.lo) < entry_extent,
                 body, [] {});
        }
        bindings_ = std::move(saved);
      };
    }
    if (const auto *op = stmt.as<WhileNode>()) {
      bool grows = false;
      std::unordered_set<const VarNode *> writes;
      PostOrderVisit(op->body, [&](const ObjectRef &node) {
        if (const auto *call = node.as<CallNode>()) {
          grows |= updates_.count(call) != 0;
          for (const auto *data : LocalWrites(call))
            writes.insert(data);
        }
        if (const auto *store = node.as<BufferStoreNode>())
          writes.insert(store->buffer->data.get());
      });
      if (grows) {
        auto body = Build(op->body);
        return [this, op, body] {
          // A scalar induction variable may make every condition decidable.
          // Reuse the numeric body summary, with a fixed iteration budget.
          for (size_t count = 0;; ++count) {
            Work();
            PrimExpr condition = analyzer_.Simplify(Resolve(op->condition));
            if (is_zero(condition))
              return;
            ICHECK(is_one(condition) && count < 4096)
                << "T.dist cannot prove cumulative expected for a "
                   "data-dependent while";
            body();
          }
        };
      }
      // A completion consumer does not increase expected. Unknown iterations
      // invalidate its auxiliary writes instead of simulating pending progress.
      return [this, writes] {
        for (const auto *data : writes)
          Reset(data);
      };
    }
    if (const auto *op = stmt.as<LetStmtNode>()) {
      auto body = Build(op->body);
      return [this, op, body] {
        auto saved = bindings_;
        if (needed_vars_.count(op->var.get()))
          bindings_[op->var.get()] =
              Snapshot(Eval(op->value), op->var.dtype()).snapshot;
        body();
        bindings_ = std::move(saved);
      };
    }
    if (const auto *op = stmt.as<BufferStoreNode>())
      return [this, op] {
        Work();
        Write(op->buffer, op->indices, Eval(op->value));
      };
    if (const auto *op = stmt.as<AllocateNode>()) {
      auto body = Build(op->body);
      return [this, op, body] {
        auto init = op->annotations.Get(tl::attr::kLocalVarInit);
        Reset(
            op->buffer_var.get(),
            init ? std::optional<Bounds>(Eval(Downcast<PrimExpr>(init.value())))
                 : std::nullopt);
        body();
      };
    }
    if (const auto *op = stmt.as<AttrStmtNode>()) {
      auto body = Build(op->body);
      return [this, op, body] {
        if (op->attr_key == tir::attr::thread_extent) {
          auto iv = Downcast<IterVar>(op->node);
          analyzer_.Bind(iv->var, Range::FromMinExtent(0, op->value), true);
        }
        body();
      };
    }
    if (const auto *op = stmt.as<DeclBufferNode>())
      return Build(op->body);
    if (const auto *op = stmt.as<AssertStmtNode>())
      return Build(op->body);
    if (const auto *op = stmt.as<BlockRealizeNode>())
      return Build(op->block);
    if (const auto *op = stmt.as<BlockNode>()) {
      auto body = Build(op->body);
      return [this, op, body] {
        if (auto init = op->annotations.Get(tl::attr::kLocalVarInit)) {
          for (const auto &[data, value] :
               Downcast<Map<Var, PrimExpr>>(init.value()))
            Reset(data.get(), Eval(value));
        }
        body();
      };
    }
    const auto *eval = stmt.as<EvaluateNode>();
    ICHECK(eval) << "Unsupported expected growth summary statement "
                 << stmt->GetTypeKey();
    const auto *call = eval->value.as<CallNode>();
    ICHECK(call);
    return [this, call] {
      Work();
      auto it = updates_.find(call);
      if (it != updates_.end())
        for (const Update &u : it->second)
          Grow(u);
      Invalidate(call);
    };
  }

  Stmt source_;
  bool enabled_, nonnegative_{true};
  size_t work_{0};
  arith::Analyzer analyzer_;
  State memory_;
  std::unordered_map<const CallNode *, std::vector<Update>> updates_;
  std::unordered_map<const VarNode *, PrimExpr> lets_, bindings_;
  std::unordered_set<const VarNode *> needed_vars_, counters_, remote_;
  std::unordered_set<const StmtNode *> relevant_;
  std::unordered_map<const VarNode *, int64_t> peaks_;
};

class InjectDistSignalAdvance : public StmtExprMutator {
public:
  explicit InjectDistSignalAdvance(
      const PrimFunc &func,
      std::unordered_set<const EvaluateNode *> empty_submits)
      : empty_submits_(std::move(empty_submits)), bounds_(func) {}

  static PrimFunc
  Run(PrimFunc func,
      std::unordered_set<const EvaluateNode *> empty_submits = {}) {
    InjectDistSignalAdvance injector(func, std::move(empty_submits));
    Stmt body = injector(func->body);
    injector.bounds_.Check();
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  PrimExpr StorageDelta(DataType dtype, const PrimExpr &delta) {
    // Fold constant integer narrowing with its ordinary modulo semantics.
    // Some generic simplifier paths cannot construct an out-of-range u32
    // literal when a u64 constant is narrowed. Validation uses the raw delta.
    std::optional<uint64_t> constant;
    if (const auto *imm = delta.as<IntImmNode>())
      constant = static_cast<uint64_t>(imm->value);
    else if (const auto *call = delta.as<CallNode>();
             call && call->op.same_as(builtin::large_uint_imm()))
      constant =
          static_cast<uint64_t>(Downcast<IntImm>(call->args[0])->value) |
          (static_cast<uint64_t>(Downcast<IntImm>(call->args[1])->value) << 32);
    if (constant && dtype.is_uint() && dtype.bits() < 64)
      return make_const(dtype,
                        constant.value() & ((uint64_t{1} << dtype.bits()) - 1));
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
      Array<PrimExpr> args = call->args;
      if (generation.dtype() != kind.state_dtype) {
        args.Set(5, Cast(kind.state_dtype, generation));
      }
      Stmt send = Evaluate(
          Call(call->dtype, call->op, args, call->annotations, call->span));
      return SeqStmt::Flatten(Array<Stmt>{advance, send});
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
      Array<PrimExpr> args = call->args;
      if (generation.dtype() != kind.state_dtype) {
        args.Set(3, Cast(kind.state_dtype, generation));
      }
      Stmt send = Evaluate(
          Call(call->dtype, call->op, args, call->annotations, call->span));
      return SeqStmt::Flatten(Array<Stmt>{advance, send});
    }
    if (call->op.same_as(dist_wait_signal_())) {
      const auto &kind =
          RequireDistSignalKindInfo(call->args[0], "signal kind");
      Array<PrimExpr> args = call->args;
      if (args[2].dtype() != kind.state_dtype) {
        args.Set(2, Cast(kind.state_dtype, args[2]));
      }
      return Evaluate(
          Call(call->dtype, call->op, args, call->annotations, call->span));
    }
    if (call->op.same_as(dist_expect_())) {
      ICHECK_EQ(call->args.size(), 2U);
      BufferLoad expected = RequireLocalState(call->args[0], "expected");
      // Preserve the original growth site before runtime storage casts. The
      // function's final mathematical flag bounds are checked after emission.
      bounds_.Register(call, expected, call->args[1]);
      PrimExpr delta = StorageDelta(expected->dtype, call->args[1]);
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
      // Pending membership is based on the logical delta, independently of
      // the signal storage representation.
      PrimExpr active = delta > make_zero(delta.dtype());
      PrimExpr signal_index =
          signal_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr completion_index =
          completion_expect->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr pending_index =
          pending->region[0]->min + IntImm(DataType::Int(32), index);
      PrimExpr current = BufferLoad(signal_expect->buffer, {signal_index});
      if (!is_zero(advance_expected))
        bounds_.Register(
            call, Downcast<BufferLoad>(current),
            Select(advance_expected, delta, make_zero(delta.dtype())));
      PrimExpr typed_delta = StorageDelta(signal_expect->buffer->dtype, delta);
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
  ExpectedFinalBounds bounds_;
};

PrimFunc Run(PrimFunc func) {
  if (!IsMultiRank(func)) {
    return func;
  }
  DistOpDetector detector(/*high_level=*/false);
  if (!detector.Detect(func->body)) {
    return func;
  }
  DistQueueValidator::Result queue = DistQueueValidator::Validate(func);
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
