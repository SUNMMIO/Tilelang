#include "codegen_sunmmio.h"

#include "../../op/builtin.h"
#include "../../op/dist_comm.h"
#include "../../op/utils.h"
#include "../../support/finite_width_integer.h"
#include "../../transform/dist_transform_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>

#include <algorithm>
#include <queue>

namespace tvm {
namespace codegen {
using namespace tir;
using namespace tl::dist_transform;

namespace {
struct DistBatchScopeFailure {
  const CallNode *evidence;
  const StmtNode *where;
  const char *reason;
};

} // namespace

void CodeGenTileLangSunMMIO::ClearDistFunctionState() {
  dist_destinations_.clear();
  dist_batch_for_call_.clear();
  dist_begins_.clear();
  dist_for_plans_.clear();
  dist_batches_.clear();
  dist_signals_.clear();
  dist_capture_order_.clear();
  dist_let_bindings_.clear();
  dist_rank_domain_.reset();
  dist_captures_.clear();
  dist_rebuild_submits_.clear();
  dist_rebuild_before_.clear();
  dist_capture_live_out_.clear();
  dist_rebuilt_batches_.clear();
}

void CodeGenTileLangSunMMIO::AnalyzeDistFunction(const PrimFunc &func) {
  ClearDistFunctionState();
  if (auto index = func->GetAttr<Integer>("tl.dist.rank_id_param_index")) {
    if (auto world = func->GetAttr<Integer>(kDistWorldSizeAttr)) {
      ICHECK_GE(index.value()->value, 0);
      ICHECK_LT(index.value()->value, func->params.size());
      dist_rank_domain_ = std::make_pair(func->params[index.value()->value],
                                         world.value()->value);
    }
  }

  class Analyzer final : public StmtExprVisitor {
  public:
    explicit Analyzer(CodeGenTileLangSunMMIO *codegen) : codegen_(codegen) {}
    void Run(const Stmt &body) {
      VisitStmt(body);
      if (current_) {
        Fail(*current_, body.get(),
             "SunMMIO dist batch has no reachable consumer");
      }
      std::vector<std::vector<const Event *>> batch_events(batches_.size());
      for (const Event &event : events_) {
        size_t id = Find(event.batch);
        codegen_->dist_batch_for_call_[event.call] = id;
        batch_events[id].push_back(&event);
      }
      for (size_t id = 0; id < batches_.size(); ++id) {
        if (Find(id) != id) {
          continue;
        }
        const Batch &batch = batches_[id];
        ICHECK(batch.consumer) << "SunMMIO dist batch is never consumed";
        const Consumer &consumer = *batch.consumer;
        int scope =
            batch.origin == Origin::kBegin ? consumer.scope : batch.scope;
        if (scope != consumer.scope) {
          Fail(id, consumer.stmt,
               "SunMMIO dist batch requires finite reconstruction across "
               "control-flow regions");
        }
        const StmtNode *first = nullptr;
        for (const Event *event : batch_events[id]) {
          int ancestor = event->scope;
          const StmtNode *anchor = event->stmt;
          while (ancestor != scope && ancestor != 0) {
            anchor = scopes_[ancestor].owner;
            ancestor = scopes_[ancestor].parent;
          }
          if (ancestor != scope) {
            Fail(id, event->stmt,
                 "SunMMIO dist submit requires finite batch reconstruction");
          }
          if (!first) {
            first = anchor;
          }
        }
        if (batch.origin == Origin::kBegin) {
          // An empty loop output is opened immediately before its yield.
          if (!first && consumer.kind != ConsumerKind::kYield) {
            first = consumer.stmt;
          }
          if (first) {
            codegen_->dist_begins_[first].push_back(id);
          }
        }
      }
      for (auto &[loop, plan] : codegen_->dist_for_plans_) {
        plan.initial = Find(plan.initial);
        plan.argument = Find(plan.argument);
        plan.yielded = Find(plan.yielded);
        plan.result = Find(plan.result);
      }
    }

  private:
    enum class Origin { kBegin, kForArgument, kForResult };
    enum class ConsumerKind { kSubmit, kFor, kYield };
    struct Consumer {
      int scope;
      const StmtNode *stmt;
      ConsumerKind kind;
    };
    struct Batch {
      size_t parent;
      Origin origin{Origin::kBegin};
      int scope{0};
      std::optional<Consumer> consumer;
    };
    struct Scope {
      int parent;
      const StmtNode *owner;
    };
    struct Event {
      const CallNode *call;
      size_t batch;
      int scope;
      const StmtNode *stmt;
    };

    size_t Find(size_t id) {
      if (batches_[id].parent != id) {
        batches_[id].parent = Find(batches_[id].parent);
      }
      return batches_[id].parent;
    }
    size_t NewBatch(Origin origin = Origin::kBegin, int scope = 0) {
      size_t id = batches_.size();
      batches_.push_back({id, origin, scope, std::nullopt});
      return id;
    }
    const CallNode *Evidence(size_t id) {
      id = Find(id);
      for (const Event &event : events_) {
        if (Find(event.batch) == id)
          return event.call;
      }
      for (const auto &[loop, plan] : codegen_->dist_for_plans_) {
        if (Find(plan.result) == id) {
          if (const CallNode *call = Evidence(plan.initial))
            return call;
          return Evidence(plan.argument);
        }
      }
      return nullptr;
    }
    [[noreturn]] void Fail(size_t id, const StmtNode *where,
                           const char *reason) {
      throw DistBatchScopeFailure{Evidence(id), where, reason};
    }
    void Consume(size_t id, Consumer consumer) {
      Batch &batch = batches_[Find(id)];
      if (batch.consumer) {
        Fail(id, consumer.stmt,
             "SunMMIO conditional submit requires finite batch reconstruction");
      }
      batch.consumer = consumer;
    }
    std::optional<size_t> Merge(std::optional<size_t> a,
                                std::optional<size_t> b) {
      if (!a)
        return b;
      if (!b)
        return a;
      size_t lhs = Find(*a), rhs = Find(*b);
      if (lhs == rhs)
        return lhs;
      if (batches_[lhs].origin != Origin::kBegin ||
          batches_[rhs].origin != Origin::kBegin || batches_[lhs].consumer ||
          batches_[rhs].consumer) {
        Fail(lhs, stmt_,
             "SunMMIO cannot merge batch handles through If/While results");
      }
      batches_[rhs].parent = lhs;
      return lhs;
    }
    void VisitStmt(const Stmt &stmt) final {
      const StmtNode *saved = stmt_;
      stmt_ = stmt.get();
      StmtExprVisitor::VisitStmt(stmt);
      stmt_ = saved;
    }
    void VisitInScope(const Stmt &body, const StmtNode *owner) {
      int saved = scope_;
      scope_ = scopes_.size();
      scopes_.push_back({saved, owner});
      VisitStmt(body);
      scope_ = saved;
    }
    void VisitStmt_(const LetStmtNode *op) final {
      if (op->var.dtype().is_int() || op->var.dtype().is_uint()) {
        codegen_->dist_let_bindings_.emplace(op->var.get(), op->value);
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const LetNode *op) final {
      if (op->var.dtype().is_int() || op->var.dtype().is_uint()) {
        codegen_->dist_let_bindings_.emplace(op->var.get(), op->value);
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const IfThenElseNode *op) final {
      VisitExpr(op->condition);
      auto incoming = current_;
      VisitInScope(op->then_case, op);
      auto then_queue = current_;
      current_ = incoming;
      if (op->else_case) {
        VisitInScope(op->else_case.value(), op);
      }
      if (incoming) {
        if (batches_[Find(*incoming)].consumer) {
          Fail(*incoming, op,
               "SunMMIO conditional submit requires finite batch "
               "reconstruction");
        }
      }
      current_ = Merge(then_queue, current_);
    }
    void VisitLoop(const Stmt &body, const StmtNode *owner,
                   const ForNode *for_loop) {
      auto incoming = current_;
      size_t first_event = events_.size();
      size_t first_submit = submits_.size();
      int parent_scope = scope_;
      int body_scope = scopes_.size();
      VisitInScope(body, owner);
      auto outgoing = current_;
      bool consumed =
          incoming && batches_[Find(*incoming)].consumer.has_value();
      if ((!incoming && !outgoing) || (incoming && !consumed) ||
          (!incoming && outgoing && first_submit == submits_.size())) {
        return;
      }
      if (!for_loop) {
        Fail(incoming ? *incoming : submits_[first_submit], owner,
             "SunMMIO cannot carry a consumed batch through While directly");
      }
      size_t initial, argument;
      if (incoming) {
        initial = Find(*incoming);
        auto consumer = batches_[initial].consumer;
        if (!consumer || consumer->scope != body_scope ||
            consumer->kind == ConsumerKind::kYield) {
          Fail(initial, owner,
               "SunMMIO cannot establish a direct For batch consumer");
        }
        argument = NewBatch(Origin::kForArgument, body_scope);
        batches_[argument].consumer = consumer;
        if (consumer->kind == ConsumerKind::kFor) {
          auto nested = codegen_->dist_for_plans_.find(
              static_cast<const ForNode *>(consumer->stmt));
          ICHECK(nested != codegen_->dist_for_plans_.end());
          nested->second.initial = argument;
        }
        batches_[initial].consumer.reset();
        for (size_t i = first_event; i < events_.size(); ++i) {
          if (Find(events_[i].batch) == initial) {
            events_[i].batch = argument;
          }
        }
      } else {
        argument = Find(submits_[first_submit]);
        const auto &consumer = batches_[argument].consumer;
        if (!consumer || consumer->scope != body_scope ||
            consumer->kind != ConsumerKind::kSubmit) {
          Fail(argument, owner,
               "SunMMIO loop submissions need finite reconstruction");
        }
        batches_[argument].origin = Origin::kForArgument;
        batches_[argument].scope = body_scope;
        initial = NewBatch();
      }
      Consume(initial, {parent_scope, owner, ConsumerKind::kFor});
      size_t yielded = outgoing ? Find(*outgoing) : NewBatch();
      Consume(yielded, {body_scope, owner, ConsumerKind::kYield});
      size_t result = NewBatch(Origin::kForResult, parent_scope);
      codegen_->dist_for_plans_.emplace(
          for_loop, DistForPlan{initial, argument, yielded, result, !outgoing});
      current_ = result;
    }
    void VisitStmt_(const ForNode *op) final {
      VisitExpr(op->min);
      VisitExpr(op->extent);
      VisitLoop(op->body, op, op);
    }
    void VisitStmt_(const WhileNode *op) final {
      VisitExpr(op->condition);
      VisitLoop(op->body, op, nullptr);
    }
    void VisitExpr_(const CallNode *op) final {
      bool put = op->op.same_as(tl::dist_put_());
      bool signal = op->op.same_as(tl::dist_signal_put_());
      bool submit = op->op.same_as(tl::dist_submit_());
      if (put) {
        BufferRegion dst = tl::NormalizeToBufferRegion(op->args[1]);
        codegen_->dist_destinations_.insert(dst->buffer->data.get());
      }
      if (codegen_->dist_captures_.count(op) ||
          codegen_->dist_rebuild_submits_.count(op)) {
        StmtExprVisitor::VisitExpr_(op);
        return;
      }
      if (put || signal || submit) {
        if (!current_)
          current_ = NewBatch();
        size_t id = Find(*current_);
        events_.push_back({op, id, scope_, stmt_});
        if (submit) {
          Consume(id, {scope_, stmt_, ConsumerKind::kSubmit});
          submits_.push_back(id);
          current_.reset();
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    CodeGenTileLangSunMMIO *codegen_;
    std::optional<size_t> current_;
    std::vector<Batch> batches_;
    std::vector<Event> events_;
    std::vector<size_t> submits_;
    std::vector<Scope> scopes_{{0, nullptr}};
    const StmtNode *stmt_{nullptr};
    int scope_{0};
  };
  while (true) {
    // Recompute direct analysis, preserving captures and reconstruction plans.
    dist_destinations_.clear();
    dist_let_bindings_.clear();
    dist_batch_for_call_.clear();
    dist_begins_.clear();
    dist_for_plans_.clear();
    try {
      Analyzer(this).Run(func->body);
      break;
    } catch (const DistBatchScopeFailure &failure) {
      ICHECK(failure.evidence)
          << failure.reason << ": " << ffi::GetRef<Stmt>(failure.where);
      PlanDistReconstruction(func, failure.evidence);
    }
  }
}

void CodeGenTileLangSunMMIO::PlanDistReconstruction(const PrimFunc &func,
                                                    const CallNode *evidence) {
  class Planner final : public StmtExprVisitor {
  public:
    explicit Planner(CodeGenTileLangSunMMIO *codegen) : codegen_(codegen) {}
    void Run(const Stmt &body, const CallNode *evidence) {
      VisitStmt(body);
      auto found = ids_.find(evidence);
      ICHECK(found != ids_.end()) << "Missing dist reconstruction evidence";
      size_t root = Find(found->second);
      std::vector<size_t> entries;
      std::vector<int> indegree(nodes_.size(), 0);
      for (size_t id = 0; id < nodes_.size(); ++id) {
        if (Find(id) != root || !nodes_[id].entry)
          continue;
        ICHECK(!nodes_[id].expired)
            << "SunMMIO dist captured root storage is unavailable at submit "
               "after allocation scope exit: "
            << ffi::GetRef<PrimExpr>(nodes_[id].call);
        ICHECK(!nodes_[id].history)
            << "SunMMIO dist reconstruction requires dynamic send history; "
               "one static put can remain pending across repeated executions: "
            << ffi::GetRef<PrimExpr>(nodes_[id].call);
        entries.push_back(id);
        for (size_t next : nodes_[id].next) {
          if (Find(next) == root)
            ++indegree[next];
        }
      }
      std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>>
          ready;
      for (size_t id : entries)
        if (!indegree[id])
          ready.push(id);
      std::vector<size_t> order;
      while (!ready.empty()) {
        size_t id = ready.top();
        ready.pop();
        order.push_back(id);
        for (size_t next : nodes_[id].next) {
          if (Find(next) == root && --indegree[next] == 0)
            ready.push(next);
        }
      }
      ICHECK_EQ(order.size(), entries.size())
          << "SunMMIO dist reconstruction cannot preserve dynamic queue order "
             "with finite capture slots: "
          << ffi::GetRef<PrimExpr>(evidence);
      bool added = false;
      for (size_t id : order) {
        const Node &node = nodes_[id];
        if (codegen_->dist_captures_.emplace(node.call, DistCapture{}).second) {
          codegen_->dist_capture_order_.push_back(node.call);
          added = true;
        }
        for (const StmtNode *control : node.controls) {
          codegen_->dist_capture_live_out_[control][node.call] = true;
        }
      }
      ICHECK(added) << "SunMMIO dist batch cannot be represented by finite "
                       "reconstruction";
      for (size_t id = 0; id < nodes_.size(); ++id) {
        const Node &node = nodes_[id];
        if (Find(id) != root || node.entry)
          continue;
        std::vector<const CallNode *> appends;
        for (size_t entry : order) {
          if (std::find(node.pending.begin(), node.pending.end(), entry) ==
              node.pending.end())
            continue;
          for (const VarNode *allocation : nodes_[entry].required_allocations) {
            ICHECK(std::find(node.allocations.begin(), node.allocations.end(),
                             allocation) != node.allocations.end())
                << "SunMMIO dist captured root storage is unavailable at "
                   "submit: "
                << ffi::GetRef<PrimExpr>(nodes_[entry].call);
          }
          appends.push_back(nodes_[entry].call);
          for (const StmtNode *control : node.controls) {
            codegen_->dist_capture_live_out_[control].emplace(
                nodes_[entry].call, false);
          }
        }
        auto inserted =
            codegen_->dist_rebuild_submits_.emplace(node.call, appends);
        if (inserted.second) {
          codegen_->dist_rebuild_before_[node.anchor].push_back(node.call);
        } else {
          inserted.first->second = std::move(appends);
        }
      }
    }

  private:
    struct Node {
      const CallNode *call;
      size_t parent;
      bool entry;
      bool history{false};
      bool expired{false};
      std::vector<size_t> pending;
      std::vector<size_t> next;
      std::vector<const StmtNode *> controls;
      std::vector<const VarNode *> allocations;
      std::vector<const VarNode *> required_allocations;
      const StmtNode *anchor{nullptr};
    };
    size_t Find(size_t id) {
      if (nodes_[id].parent != id)
        nodes_[id].parent = Find(nodes_[id].parent);
      return nodes_[id].parent;
    }
    static std::vector<size_t> Merge(std::vector<size_t> a,
                                     const std::vector<size_t> &b) {
      a.insert(a.end(), b.begin(), b.end());
      std::sort(a.begin(), a.end());
      a.erase(std::unique(a.begin(), a.end()), a.end());
      return a;
    }
    size_t GetNode(const CallNode *call, bool entry) {
      auto old = ids_.find(call);
      if (old != ids_.end())
        return old->second;
      size_t id = nodes_.size();
      ids_.emplace(call, id);
      nodes_.push_back({call, id, entry});
      Node &node = nodes_.back();
      node.controls = controls_;
      node.allocations = allocations_;
      node.anchor = stmt_;
      auto anchor = anchors_.find(call);
      if (anchor != anchors_.end())
        node.anchor = anchor->second;
      if (call->op.same_as(tl::dist_put_())) {
        for (size_t i : {size_t{0}, size_t{1}}) {
          Buffer region = tl::NormalizeToBufferRegion(call->args[i])->buffer;
          if (std::find(allocations_.begin(), allocations_.end(),
                        region->data.get()) != allocations_.end()) {
            node.required_allocations.push_back(region->data.get());
          }
        }
      }
      return id;
    }
    void VisitStmt(const Stmt &stmt) final {
      const StmtNode *old = stmt_;
      stmt_ = stmt.get();
      StmtExprVisitor::VisitStmt(stmt);
      stmt_ = old;
    }
    void VisitStmt_(const SeqStmtNode *op) final {
      for (size_t i = 0; i < op->seq.size(); ++i) {
        const auto *evaluate = op->seq[i].as<EvaluateNode>();
        const auto *submit =
            evaluate ? evaluate->value.as<CallNode>() : nullptr;
        if (submit && submit->op.same_as(tl::dist_submit_()) && i > 0) {
          const auto *previous = op->seq[i - 1].as<EvaluateNode>();
          const auto *wait =
              previous ? previous->value.as<CallNode>() : nullptr;
          if (wait && wait->op.same_as(tl::dist_wait_send())) {
            anchors_[submit] = previous;
          }
        }
        VisitStmt(op->seq[i]);
      }
    }
    void VisitStmt_(const AllocateNode *op) final {
      allocations_.push_back(op->buffer_var.get());
      StmtExprVisitor::VisitStmt_(op);
      for (size_t entry : pending_) {
        auto &node = nodes_[entry];
        if (std::find(node.required_allocations.begin(),
                      node.required_allocations.end(), op->buffer_var.get()) !=
            node.required_allocations.end()) {
          node.expired = true;
        }
      }
      allocations_.pop_back();
    }
    void VisitStmt_(const IfThenElseNode *op) final {
      VisitExpr(op->condition);
      auto incoming = pending_;
      controls_.push_back(op);
      VisitStmt(op->then_case);
      auto then_queue = pending_;
      pending_ = incoming;
      if (op->else_case)
        VisitStmt(op->else_case.value());
      pending_ = Merge(std::move(then_queue), pending_);
      controls_.pop_back();
    }
    void VisitLoop(const Stmt &body, const StmtNode *loop) {
      auto incoming = pending_, entry = incoming;
      controls_.push_back(loop);
      while (true) {
        pending_ = entry;
        VisitStmt(body);
        auto next = Merge(incoming, pending_);
        if (next == entry)
          break;
        entry = std::move(next);
      }
      pending_ = Merge(incoming, pending_);
      controls_.pop_back();
    }
    void VisitStmt_(const ForNode *op) final { VisitLoop(op->body, op); }
    void VisitStmt_(const WhileNode *op) final { VisitLoop(op->body, op); }
    void VisitExpr_(const CallNode *op) final {
      bool put = op->op.same_as(tl::dist_put_()) ||
                 op->op.same_as(tl::dist_signal_put_());
      bool submit = op->op.same_as(tl::dist_submit_());
      if (put) {
        size_t id = GetNode(op, true);
        for (size_t previous : pending_) {
          if (previous == id) {
            nodes_[id].history = true;
          } else {
            auto &edges = nodes_[previous].next;
            if (std::find(edges.begin(), edges.end(), id) == edges.end())
              edges.push_back(id);
          }
        }
        pending_ = Merge(std::move(pending_), {id});
      } else if (submit) {
        size_t id = GetNode(op, false);
        nodes_[id].pending = Merge(std::move(nodes_[id].pending), pending_);
        for (size_t entry : pending_)
          nodes_[Find(entry)].parent = Find(id);
        pending_.clear();
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    CodeGenTileLangSunMMIO *codegen_;
    std::vector<Node> nodes_;
    std::unordered_map<const CallNode *, size_t> ids_;
    std::unordered_map<const CallNode *, const StmtNode *> anchors_;
    std::vector<size_t> pending_;
    std::vector<const StmtNode *> controls_;
    std::vector<const VarNode *> allocations_;
    const StmtNode *stmt_{nullptr};
  };
  Planner(this).Run(func->body, evidence);
}

void CodeGenTileLangSunMMIO::BindDistCaptures() {
  auto field = [&](const SunMMIOType &type) {
    return builder_->ConstantInt(NewValueName(), 0, type, type.dtype);
  };
  SunMMIOType index_type{SunMMIOType::Kind::kIndex, DataType::Int(32), 1, {}};
  for (const CallNode *call : dist_capture_order_) {
    DistCapture &capture = dist_captures_.at(call);
    bool put = call->op.same_as(tl::dist_put_());
    const auto &kind =
        RequireDistSignalKindInfo(call->args[put ? 3 : 1], "signal kind");
    capture.valid = field(MapType(DataType::Bool()));
    capture.peer = field(MapType(DataType::Int(32)));
    capture.index = field(index_type);
    if (kind.update_mode != DistSignalUpdateMode::kIncrement) {
      capture.generation = field(MapType(kind.state_dtype));
    }
    if (put) {
      capture.src = tl::NormalizeToBufferRegion(call->args[0]);
      capture.dst = tl::NormalizeToBufferRegion(call->args[1]);
      for (size_t i = 0; i < capture.src->get()->region.size(); ++i)
        capture.src_indices.push_back(field(index_type));
      for (size_t i = 0; i < capture.dst->get()->region.size(); ++i)
        capture.dst_indices.push_back(field(index_type));
    }
  }
}

void CodeGenTileLangSunMMIO::AppendDistCaptureLiveOut(
    const StmtNode *stmt, std::vector<SunMMIOValue> *live_out) {
  auto changes = dist_capture_live_out_.find(stmt);
  if (changes == dist_capture_live_out_.end())
    return;
  for (const CallNode *call : dist_capture_order_) {
    auto changed = changes->second.find(call);
    if (changed == changes->second.end())
      continue;
    const DistCapture &capture = dist_captures_.at(call);
    live_out->push_back(capture.valid);
    if (!changed->second)
      continue;
    live_out->push_back(capture.peer);
    live_out->push_back(capture.index);
    if (capture.generation)
      live_out->push_back(*capture.generation);
    live_out->insert(live_out->end(), capture.src_indices.begin(),
                     capture.src_indices.end());
    live_out->insert(live_out->end(), capture.dst_indices.begin(),
                     capture.dst_indices.end());
  }
}

void CodeGenTileLangSunMMIO::EmitDistCapture(const CallNode *op) {
  DistCapture &capture = dist_captures_.at(op);
  bool put = op->op.same_as(tl::dist_put_());
  size_t pos = put ? 3 : 1;
  MarkVisitedExprTree(op->args[pos]);
  auto save = [&](const SunMMIOValue &slot, const SunMMIOValue &value) {
    builder_->BindValueAlias(slot.value, value);
  };
  save(capture.peer, EnsureType(EvalExpr(op->args[put ? 2 : 0]),
                                MapType(DataType::Int(32)), DataType::Int(32)));
  if (put) {
    auto region = [&](const PrimExpr &expr,
                      const std::vector<SunMMIOValue> &slots) {
      BufferRegion view = NormalizeRegionTracked(expr);
      CheckDistRegionAlignment(view);
      // Let bindings prove alignment; evaluating their expanded expressions
      // would repeat memory reads instead of using the original SSA snapshot.
      for (size_t dim = 0; dim < view->region.size(); ++dim) {
        const Range &range = view->region[dim];
        MarkVisitedExprRoot(range->extent);
        save(slots[dim],
             EmitRegionPartitionIndex(range->min,
                                      Downcast<IntImm>(range->extent)->value));
      }
    };
    region(op->args[0], capture.src_indices);
    region(op->args[1], capture.dst_indices);
  }
  save(capture.index, EnsureIndex(EvalExpr(op->args[pos + 1])));
  if (capture.generation)
    save(*capture.generation, EvalExpr(op->args[pos + 2]));
  else
    MarkVisitedExprTree(op->args[pos + 2]);
  save(capture.valid,
       builder_->ConstantInt(NewValueName(), 1, MapType(DataType::Bool()),
                             DataType::Bool()));
}

void CodeGenTileLangSunMMIO::PrepareDistRebuilds(const StmtNode *stmt) {
  auto submits = dist_rebuild_before_.find(stmt);
  if (submits == dist_rebuild_before_.end())
    return;
  for (const CallNode *submit : submits->second) {
    SunMMIOValue batch = builder_->DistBatchBegin(NewValueName());
    for (const CallNode *call : dist_rebuild_submits_.at(submit)) {
      const DistCapture &capture = dist_captures_.at(call);
      bool put = call->op.same_as(tl::dist_put_());
      const auto &kind =
          RequireDistSignalKindInfo(call->args[put ? 3 : 1], "signal kind");
      builder_->BeginIf(capture.valid, {});
      if (put) {
        auto view = [&](const BufferRegion &region,
                        const std::vector<SunMMIOValue> &indices) {
          const BufferBinding &root = LookupBuffer(region->buffer);
          std::vector<int64_t> extents;
          for (const Range &range : region->region)
            extents.push_back(range->extent.as<IntImmNode>()->value);
          return builder_->RegionCall(NewValueName(), root.handle, indices,
                                      extents, DataType::Handle(),
                                      MapType(DataType::Handle()));
        };
        SunMMIOValue src = view(*capture.src, capture.src_indices);
        SunMMIOValue dst = view(*capture.dst, capture.dst_indices);
        builder_->DistBatchPut(batch, capture.peer, src, dst);
      }
      builder_->DistBatchSignal(batch, capture.peer,
                                dist_signals_.at(kind.name), capture.index,
                                capture.generation);
      builder_->EndIf();
    }
    dist_rebuilt_batches_.emplace(submit, batch);
  }
}

void CodeGenTileLangSunMMIO::EmitDistBatchBegins(const StmtNode *stmt) {
  auto begins = dist_begins_.find(stmt);
  if (begins == dist_begins_.end())
    return;
  for (size_t id : begins->second) {
    dist_batches_[id] = builder_->DistBatchBegin(NewValueName());
  }
}

void CodeGenTileLangSunMMIO::PrepareDistFor(
    const ForNode *op, std::vector<SunMMIOValue> *live_out) {
  auto plan = dist_for_plans_.find(op);
  if (plan == dist_for_plans_.end())
    return;
  SunMMIOValue initial = dist_batches_.at(plan->second.initial);
  live_out->push_back(initial);
  dist_batches_[plan->second.argument] = initial;
}

void CodeGenTileLangSunMMIO::YieldDistFor(const ForNode *op) {
  auto plan = dist_for_plans_.find(op);
  if (plan == dist_for_plans_.end())
    return;
  size_t yielded = plan->second.yielded;
  if (plan->second.empty_yield) {
    dist_batches_[yielded] = builder_->DistBatchBegin(NewValueName());
  }
  ICHECK(dist_batches_.count(yielded))
      << "Missing SunMMIO For batch output binding";
  const auto &initial = dist_batches_.at(plan->second.initial);
  builder_->BindValueAlias(initial.value, dist_batches_.at(yielded));
}

void CodeGenTileLangSunMMIO::FinishDistFor(const ForNode *op) {
  auto plan = dist_for_plans_.find(op);
  if (plan == dist_for_plans_.end())
    return;
  SunMMIOValue result = dist_batches_.at(plan->second.initial);
  dist_batches_.erase(plan->second.initial);
  dist_batches_.erase(plan->second.argument);
  dist_batches_.erase(plan->second.yielded);
  dist_batches_[plan->second.result] = result;
}

void CodeGenTileLangSunMMIO::BindDistResources(const PrimFunc &func) {
  auto counts =
      func->GetAttr<ffi::Map<ffi::String, Integer>>(kDistSignalCountsAttr);
  if (!counts) {
    return;
  }
  for (const auto &kind : DistSignalKindInfos()) {
    auto count = counts.value().Get(kind.name);
    if (!count || count.value()->value == 0) {
      continue;
    }
    ICHECK(kind.kind != DistSignalKind::kDramMemory)
        << "SunMMIO DRAM MEMORY signal external resource ABI is not "
           "implemented";
    SunMMIOType type;
    type.kind = SunMMIOType::Kind::kDistSignal;
    type.dtype = DataType::Handle();
    type.signal_kind =
        kind.update_mode == DistSignalUpdateMode::kIncrement ? "increment"
        : kind.update_mode == DistSignalUpdateMode::kValue   ? "assign"
                                                             : "memory";
    type.memory_scope =
        kind.scope == DistSignalScope::kSram ? "shared.rsram" : "global";
    type.signal_count = count.value()->value;
    dist_signals_.emplace(kind.name,
                          builder_->DistSignal(NewValueName(), type));
  }
}

void CodeGenTileLangSunMMIO::CheckDistRegionAlignment(
    const BufferRegion &region) const {
  for (const Range &range : region->region) {
    const auto *extent = range->extent.as<IntImmNode>();
    ICHECK(extent && extent->value > 0)
        << "SunMMIO dist region extent must be a positive constant";
    ICHECK(CanProveRegionAlignment(range->min, extent->value))
        << "SunMMIO dist region start must be provably aligned to its extent "
           "for exact partition addressing: "
        << region;
  }
}

bool CodeGenTileLangSunMMIO::CanProveRegionAlignment(const PrimExpr &min,
                                                     int64_t extent) const {
  if (extent <= 0)
    return false;
  arith::Analyzer analyzer;
  if (dist_rank_domain_)
    analyzer.Bind(
        dist_rank_domain_->first,
        Range::FromMinExtent(make_zero(dist_rank_domain_->first.dtype()),
                             make_const(dist_rank_domain_->first.dtype(),
                                        dist_rank_domain_->second)));
  tl::FiniteWidthInteger facts(&analyzer, extent, &dist_let_bindings_);
  return facts.ProvesMultiple(min);
}

SunMMIOValue
CodeGenTileLangSunMMIO::EmitRegionPartitionIndex(const PrimExpr &min,
                                                 int64_t extent) {
  ICHECK_GT(extent, 0);
  ICHECK(CanProveRegionAlignment(min, extent))
      << "SunMMIO dist region start must be provably aligned to region extent "
      << extent << " for exact partition addressing: " << min;
  arith::Analyzer analyzer;
  if (dist_rank_domain_)
    analyzer.Bind(
        dist_rank_domain_->first,
        Range::FromMinExtent(make_zero(dist_rank_domain_->first.dtype()),
                             make_const(dist_rank_domain_->first.dtype(),
                                        dist_rank_domain_->second)));
  tl::FiniteWidthInteger facts(&analyzer, 1, &dist_let_bindings_);
  auto value = facts.Analyze(min);
  ICHECK(value.valid &&
         value.range.hi / extent <= (tl::FiniteWidthInteger::Wide(1) << 63) - 1)
      << "SunMMIO region index cannot be represented without loss";
  if (min.as<IntImmNode>()) {
    MarkVisitedExprTree(min);
    return EmitConstIndex(static_cast<int64_t>(value.range.lo / extent));
  }
  // Evaluate the original value once, before any address quantization. The
  // abstract expression is proof-only and must never be emitted or reloaded.
  MarkVisitedExprTree(min);
  SunMMIOValue start = EvalExpr(min);
  DataType dtype =
      min.dtype().is_uint() ? DataType::UInt(64) : DataType::Int(64);
  start = EmitCast(start, dtype);
  SunMMIOType type = MapType(dtype);
  SunMMIOValue divisor =
      builder_->ConstantInt(NewValueName(), extent, type, dtype);
  SunMMIOValue quotient =
      builder_->Binary(NewValueName(), BinaryOp::kDiv,
                       dtype.is_uint() ? ArithmeticFlavor::kUnsignedInt
                                       : ArithmeticFlavor::kSignedInt,
                       start, divisor, type, dtype);
  return EnsureIndex(quotient);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitDistRegion(const PrimExpr &expr) {
  CheckDistRegionAlignment(tl::NormalizeToBufferRegion(expr));
  return EmitRegionCall(expr);
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitDistWaitAny(const CallNode *op) {
  const auto &kind = RequireDistSignalKindInfo(op->args[0], "signal kind");
  MarkVisitedExprTree(op->args[0]);
  BufferRegion snapshot = NormalizeRegionTracked(op->args[3]);
  BufferRegion pending = NormalizeRegionTracked(op->args[4]);
  MarkVisitedExprTree(op->args[5]);
  EnterScope();
  Var snapshot_origin("dist_snapshot_origin", snapshot->region[0]->min.dtype());
  Var pending_origin("dist_pending_origin", pending->region[0]->min.dtype());
  BindVar(snapshot_origin, EvalExpr(snapshot->region[0]->min));
  BindVar(pending_origin, EvalExpr(pending->region[0]->min));
  SunMMIOValue base = EnsureIndex(EvalExpr(op->args[1]));
  SunMMIOType i32 = MapType(DataType::Int(32));
  SunMMIOValue count =
      EnsureType(EvalExpr(op->args[2]), i32, DataType::Int(32));
  auto constant = [&](int64_t value) {
    return builder_->ConstantInt(NewValueName(), value, i32, DataType::Int(32));
  };
  SunMMIOValue zero = constant(0), one = constant(1);
  SunMMIOValue selected = constant(-1);
  builder_->BeginWhile({selected});
  SunMMIOValue searching =
      builder_->Compare(NewValueName(), CompareOp::kLT,
                        CompareDomain::kSignedInt, selected, zero, i32);
  builder_->BeginWhileBody(searching);
  SunMMIOValue cursor = constant(0);
  Var member("dist_wait_any_member", DataType::Int(32));
  BindVar(member, cursor);
  builder_->BeginWhile({selected, cursor});
  SunMMIOValue within_count =
      builder_->Compare(NewValueName(), CompareOp::kLT,
                        CompareDomain::kSignedInt, cursor, count, i32);
  searching = builder_->Compare(NewValueName(), CompareOp::kLT,
                                CompareDomain::kSignedInt, selected, zero, i32);
  SunMMIOValue scan = builder_->Binary(
      NewValueName(), BinaryOp::kAnd, ArithmeticFlavor::kBool, within_count,
      searching, MapType(DataType::Bool()), DataType::Bool());
  builder_->BeginWhileBody(scan);
  BufferLoad pending_load(
      pending->buffer, {pending_origin + Cast(pending_origin.dtype(), member)});
  SunMMIOValue pending_value = EmitScalarTilePick(pending_load.get());
  SunMMIOValue active =
      builder_->Compare(NewValueName(), CompareOp::kNE,
                        CompareDomain::kSignedInt, pending_value, zero, i32);
  builder_->BeginIf(active, {selected});
  SunMMIOValue index =
      builder_->Binary(NewValueName(), BinaryOp::kAdd, ArithmeticFlavor::kIndex,
                       base, EnsureIndex(cursor), base.type, base.dtype);
  SunMMIOValue observed = builder_->DistSignalFetch(
      NewValueName(), dist_signals_.at(kind.name), index, kind.state_dtype);
  BufferLoad snapshot_load(
      snapshot->buffer,
      {snapshot_origin + Cast(snapshot_origin.dtype(), member)});
  SunMMIOValue expected = EmitScalarTilePick(snapshot_load.get());
  DataType comparison_dtype = kind.update_mode == DistSignalUpdateMode::kMemory
                                  ? DataType::UInt(64)
                                  : DataType::UInt(32);
  observed = EmitCast(observed, comparison_dtype);
  expected = EmitCast(expected, comparison_dtype);
  SunMMIOValue ready = builder_->Compare(NewValueName(), CompareOp::kGE,
                                         CompareDomain::kUnsignedInt, observed,
                                         expected, MapType(comparison_dtype));
  builder_->BeginIf(ready, {selected});
  builder_->BindValueAlias(selected.value, cursor);
  builder_->EndIf();
  builder_->EndIf();
  SunMMIOValue next = builder_->Binary(NewValueName(), BinaryOp::kAdd,
                                       ArithmeticFlavor::kSignedInt, cursor,
                                       one, i32, DataType::Int(32));
  builder_->BindValueAlias(cursor.value, next);
  builder_->EndWhile();
  builder_->EndWhile();
  ExitScope();
  return selected;
}

SunMMIOValue CodeGenTileLangSunMMIO::EmitDistCall(const CallNode *op) {
  SunMMIOValue result{DataType::Void(), "", {}};
  if (op->op.same_as(tl::dist_wait_send())) {
    builder_->DistFlush();
    return result;
  }
  if (op->op.same_as(tl::dist_wait_any_())) {
    return EmitDistWaitAny(op);
  }
  if (op->op.same_as(tl::dist_wait_signal_())) {
    const auto &kind = RequireDistSignalKindInfo(op->args[0], "signal kind");
    MarkVisitedExprTree(op->args[0]);
    builder_->DistSignalWait(dist_signals_.at(kind.name),
                             EnsureIndex(EvalExpr(op->args[1])),
                             EvalExpr(op->args[2]));
    return result;
  }
  if (dist_captures_.count(op)) {
    EmitDistCapture(op);
    return result;
  }
  auto rebuilt = dist_rebuild_submits_.find(op);
  if (rebuilt != dist_rebuild_submits_.end()) {
    auto batch = dist_rebuilt_batches_.find(op);
    ICHECK(batch != dist_rebuilt_batches_.end())
        << "Missing reconstructed dist batch";
    builder_->DistBatchSubmit(batch->second);
    dist_rebuilt_batches_.erase(batch);
    for (const CallNode *call : rebuilt->second) {
      const SunMMIOValue &valid = dist_captures_.at(call).valid;
      auto zero =
          builder_->ConstantInt(NewValueName(), 0, valid.type, valid.dtype);
      builder_->BindValueAlias(valid.value, zero);
    }
    return result;
  }
  auto entry = dist_batch_for_call_.find(op);
  ICHECK(entry != dist_batch_for_call_.end())
      << "Missing SunMMIO dist batch plan";
  size_t id = entry->second;
  const auto &batch = dist_batches_.at(id);
  if (op->op.same_as(tl::dist_submit_())) {
    builder_->DistBatchSubmit(batch);
    dist_batches_.erase(id);
    return result;
  }
  bool put = op->op.same_as(tl::dist_put_());
  size_t kind_pos = put ? 3 : 1;
  const auto &kind =
      RequireDistSignalKindInfo(op->args[kind_pos], "signal kind");
  MarkVisitedExprTree(op->args[kind_pos]);
  SunMMIOValue peer = EnsureType(EvalExpr(op->args[put ? 2 : 0]),
                                 MapType(DataType::Int(32)), DataType::Int(32));
  if (put) {
    SunMMIOValue src = EmitDistRegion(op->args[0]);
    SunMMIOValue dst = EmitDistRegion(op->args[1]);
    builder_->DistBatchPut(batch, peer, src, dst);
  }
  SunMMIOValue index = EnsureIndex(EvalExpr(op->args[kind_pos + 1]));
  std::optional<SunMMIOValue> generation;
  if (kind.update_mode != DistSignalUpdateMode::kIncrement) {
    generation = EvalExpr(op->args[kind_pos + 2]);
  } else {
    MarkVisitedExprTree(op->args[kind_pos + 2]);
  }
  builder_->DistBatchSignal(batch, peer, dist_signals_.at(kind.name), index,
                            generation);
  return result;
}

} // namespace codegen
} // namespace tvm
