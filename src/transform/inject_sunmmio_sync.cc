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
#include <map>
#include <set>
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

// Visitor to collect all buffer read and write accesses within an expression or
// statement. This is used to identify what memory is being touched.
class BufferAccessCollector : public ExprVisitor {
public:
  BufferAccessCollector(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

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
    auto load_region = BufferRegion(load_buffer, region);
    reads_.push_back(load_region);
  }

  void VisitExpr_(const CallNode *op) final {
    auto args = op->args;
    if (op->op.same_as(builtin::address_of())) {
      BufferRegion buffer_region;
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        buffer_region = BufferRegion::FullRegion(load->buffer);
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
    } else {
      ExprVisitor::VisitExpr_(op);
    }
  }

private:
  Array<BufferRegion> reads_;
  Map<Var, Buffer> buffer_data_to_buffer_;
};

// Collector for asynchronous operations within a loop body.
// Identifies DMA copies, MMA operations, and Broadcasts that happen
// asynchronously.
struct AccessRecord {
  Buffer buffer;
  Region region;
  bool is_read{false};
  bool is_write{false};
};

struct AsyncOpRecord {
  const EvaluateNode *op{nullptr};
  const CallNode *call{nullptr};
  int token{-1};
  int order{-1};
  std::vector<AccessRecord> reads;
  std::vector<AccessRecord> writes;
};

class LoopAsyncCollector : public StmtVisitor {
public:
  void VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = op->value.as<CallNode>();
    if (call) {
      AsyncOpRecord rec;
      rec.op = op;
      rec.call = call;
      rec.order = order_++;
      if (call->op.same_as(dma_copy())) {
        auto src = NormalizeToBufferRegion(call->args[0]);
        auto dst = NormalizeToBufferRegion(call->args[1]);
        rec.reads.push_back({src->buffer, src->region, true, false});
        rec.writes.push_back({dst->buffer, dst->region, false, true});
        async_ops.push_back(rec);
      } else if (call->op.same_as(mma_sunmmio())) {
        auto lhs = NormalizeToBufferRegion(call->args[0]);
        auto rhs = NormalizeToBufferRegion(call->args[1]);
        auto acc = NormalizeToBufferRegion(call->args[2]);
        rec.reads.push_back({lhs->buffer, lhs->region, true, false});
        rec.reads.push_back({rhs->buffer, rhs->region, true, false});
        rec.reads.push_back({acc->buffer, acc->region, true, false});
        rec.writes.push_back({acc->buffer, acc->region, false, true});
        async_ops.push_back(rec);
      } else if (call->op.same_as(broadcast_())) {
        auto src = NormalizeToBufferRegion(call->args[0]);
        auto dst = NormalizeToBufferRegion(call->args[1]);
        rec.reads.push_back({src->buffer, src->region, true, false});
        rec.writes.push_back({dst->buffer, dst->region, false, true});
        async_ops.push_back(rec);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  std::vector<AsyncOpRecord> async_ops;

private:
  int order_{0};
};

// Represents the scope of a loop for dependency tracking.
// Stores writes that happen within the loop to check for loop-carried
// dependencies.
struct LoopScope {
  Var loop_var;
  PrimExpr loop_min;
  PrimExpr loop_extent;
  std::vector<AsyncOpRecord> async_ops;
  std::map<int, std::set<int>> prev_iter_waits_by_curr_token;
  std::set<int> loop_entry_null_tokens;
  std::map<int, const CallNode *> token_to_call;
};

// Main rewriter class to inject synchronization primitives.
// It tracks buffer accesses and inserts wait_token and barrier_wait calls
// to enforce correct ordering based on data dependencies.
class InjectSyncRewriter : public StmtMutator {
public:
  InjectSyncRewriter(Map<Var, Buffer> buffer_data_to_buffer, int mesh_nrow,
                     int mesh_ncol, arith::Analyzer *analyzer)
      : buffer_data_to_buffer_(buffer_data_to_buffer), mesh_nrow_(mesh_nrow),
        mesh_ncol_(mesh_ncol), analyzer_(analyzer) {
    token_count = 0;
    barrier_count = 0;
  }

  Map<int, int> get_barrier_to_token_map() const {
    return barrier_to_token_map;
  }

  Map<int, int> get_token_to_barrier_map() const {
    return token_to_barrier_map;
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
    if (analyzer_->CanProve(scope.loop_extent <= 1)) {
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
    return false;
  }

  void AnalyzeLoopCarriedDependencies(LoopScope *scope) {
    if (!scope->loop_var.defined()) {
      return;
    }
    if (analyzer_->CanProve(scope->loop_extent <= 1)) {
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
  }

  void InjectLoopEntryNullTokens(const LoopScope &scope, Array<Stmt> &stmts) {
    for (int token : scope.loop_entry_null_tokens) {
      stmts.push_back(Evaluate(Call(DataType::Handle(), sync_null_token(),
                                    {IntImm(DataType::Int(32), token)})));
      if (token_to_barrier_map.find(token) != token_to_barrier_map.end()) {
        int barrier_id = token_to_barrier_map[token];
        stmts.push_back(
            Evaluate(Call(DataType::Handle(), barrier_init(),
                          {IntImm(DataType::Int(32), barrier_id)})));
      }
    }
  }

  // Inserts wait_token and optional barrier_wait instructions.
  // If the token is associated with a barrier (e.g. from broadcast),
  // we also need to wait on that barrier.
  void process_wait_token_and_barrier_wait(Array<Stmt> &stmts, int token_id) {
    stmts.push_back(Evaluate(Call(DataType::Handle(), wait_token(),
                                  {IntImm(DataType::Int(32), token_id)})));
    // If the current token has a corresponding barrier, we need to wait for the
    // barrier.
    if (token_to_barrier_map.find(token_id) != token_to_barrier_map.end()) {
      int barrier_id = token_to_barrier_map[token_id];
      stmts.push_back(
          Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                        {IntImm(DataType::Int(32), barrier_id)})));
    }
  }

  void InjectLoopCarriedWaitsForToken(Array<Stmt> &stmts, int curr_token_id) {
    std::unordered_set<int> injected_tokens;
    for (int i = loop_scopes_.size() - 1; i >= 0; --i) {
      auto it =
          loop_scopes_[i].prev_iter_waits_by_curr_token.find(curr_token_id);
      if (it == loop_scopes_[i].prev_iter_waits_by_curr_token.end()) {
        continue;
      }
      for (int token_id : it->second) {
        if (injected_tokens.count(token_id) != 0) {
          continue;
        }
        process_wait_token_and_barrier_wait(stmts, token_id);
        injected_tokens.insert(token_id);
      }
    }
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
          process_wait_token_and_barrier_wait(stmts, token);
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
          process_wait_token_and_barrier_wait(stmts, token);
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
          process_wait_token_and_barrier_wait(stmts, token);
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
  }

  // Helper to construct and inject a barrier_init call.
  // Also establishes the mappings between the generated token and barrier IDs.
  void init_barrier_(Array<Stmt> &stmts, int barrier_id, int token_id,
                     PrimExpr read_core, Array<PrimExpr> write_cores = {}) {
    Array<PrimExpr> args;
    args.push_back(barrier_id);
    args.push_back(read_core);
    if (!write_cores.empty()) {
      for (const auto &core : write_cores) {
        if (!analyzer_->CanProve(core == read_core)) {
          args.push_back(core);
        }
      }
    }

    stmts.push_back(Evaluate(Call(DataType::Handle(), barrier_init(), args)));

    token_to_barrier_map.Set(token_id, barrier_id);
    barrier_to_token_map.Set(barrier_id, token_id);
  }

  // Analyzes a broadcast operation and initializes a barrier for it.
  // Calculates the read core and write cores based on the mesh topology
  // (rows/cols) and the broadcast direction (horizontal or vertical),
  // considering given masks.
  void process_broadcast_barrier(const CallNode *call, int curr_token_id,
                                 int curr_barrier_id, Array<Stmt> &stmts) {
    PrimExpr src_core = call->args[3];
    int direction = call->args[4].as<IntImm>().value()->value;
    Array<int> masks;
    for (size_t i = 5; i < call->args.size(); i++) {
      masks.push_back(call->args[i].as<IntImm>().value()->value);
    }

    PrimExpr src_core_row =
        analyzer_->Simplify(tvm::floordiv(src_core, mesh_ncol_));
    PrimExpr src_core_col =
        analyzer_->Simplify(tvm::floormod(src_core, mesh_ncol_));
    auto read_cores = Array<PrimExpr>{src_core};
    Array<PrimExpr> write_cores;
    bool mask_flag = false;
    if (direction == 0) { // horizontal
      for (int j = 0; j < mesh_ncol_; j++) {
        for (const auto &mask : masks) {
          if (mask == j) {
            mask_flag = true;
            break;
          }
        }
        if (mask_flag) {
          mask_flag = false;
          continue;
        }
        write_cores.push_back(
            analyzer_->Simplify(src_core_row * mesh_ncol_ + j));
      }
    } else if (direction == 1) { // vertical
      for (int i = 0; i < mesh_nrow_; i++) {
        for (const auto &mask : masks) {
          if (mask == i) {
            mask_flag = true;
            break;
          }
        }
        if (mask_flag) {
          mask_flag = false;
          continue;
        }
        write_cores.push_back(
            analyzer_->Simplify(i * mesh_ncol_ + src_core_col));
      }
    }

    init_barrier_(stmts, curr_barrier_id, curr_token_id, src_core, write_cores);
  }

  // Extracts all buffer read and write accesses from a primitive expression
  // and processes their dependencies to inject necessary synchronization
  // tokens.
  void token_process_prim_expr(const PrimExpr &expr, Array<Stmt> &stmts) {
    auto buf_load_collector = BufferAccessCollector(buffer_data_to_buffer_);
    buf_load_collector(expr);
    Array<BufferRegion> read_regions = buf_load_collector.GetReads();
    for (const auto &read_region : read_regions) {
      token_process_read_buffer(read_region, stmts, -1, false);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode *op) {
    Array<Stmt> stmts;
    token_process_prim_expr(op->value, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const LetStmtNode *op) {
    Array<Stmt> stmts;
    token_process_prim_expr(op->value, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const WhileNode *op) {
    Array<Stmt> stmts;
    token_process_prim_expr(op->condition, stmts);

    LoopAsyncCollector collector;
    collector(op->body);

    LoopScope scope;
    scope.async_ops = collector.async_ops;
    for (auto &async_op : scope.async_ops) {
      // Pre-assign a stable token id for each async site in this loop.
      // This lets the body rewriter attach the same token id every iteration,
      // enabling consistent loop-carried dependency reasoning.
      int token = GetNextTokenId();
      async_op.token = token;
      pre_assigned_tokens_[async_op.op] = token;

      // Keep a back-reference from token -> call for special handling after we
      // finish rewriting the loop (e.g. broadcast barrier initialization).
      const CallNode *call = async_op.call;
      scope.token_to_call[token] = call;

      // For broadcast, we also need a barrier id to synchronize the data
      // movement across cores. The barrier init may be emitted later (after we
      // know whether the broadcast token is actually waited on).
      if (call && call->op.same_as(broadcast_())) {
        int barrier = GetNextBarrierId();
        token_to_barrier_map.Set(token, barrier);
        barrier_to_token_map.Set(barrier, token);
      }
    }

    AnalyzeWhileLoopCarriedDependencies(&scope);

    // Push this loop scope so nested visitors can consult it when analyzing
    // read/write accesses inside the loop body.
    loop_scopes_.push_back(scope);

    Stmt loop_stmt = StmtMutator::VisitStmt_(op);

    scope = loop_scopes_.back();
    loop_scopes_.pop_back();
    for (const auto &async_op : scope.async_ops) {
      pre_assigned_tokens_.erase(async_op.op);
    }

    InjectLoopEntryNullTokens(scope, stmts);
    stmts.push_back(loop_stmt);
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AllocateNode *op) {
    Array<Stmt> stmts;
    token_process_prim_expr(op->condition, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const BufferRealizeNode *op) {
    Array<Stmt> stmts;
    token_process_prim_expr(op->condition, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const AssertStmtNode *op) {
    Array<Stmt> stmts;
    token_process_prim_expr(op->condition, stmts);
    token_process_prim_expr(op->message, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) {
    Array<Stmt> stmts;
    token_process_prim_expr(op->predicate, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) {
    Array<Stmt> stmts;

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
    auto store_region = BufferRegion(store_buffer, region);
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
        int curr_barrier_id;
        if (token_to_barrier_map.count(curr_token_id)) {
          curr_barrier_id = token_to_barrier_map[curr_token_id];
        } else {
          curr_barrier_id = GetNextBarrierId();
        }

        InjectLoopCarriedWaitsForToken(stmts, curr_token_id);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[0]), stmts,
                                  curr_token_id);
        token_process_write_buffer(NormalizeToBufferRegion(call->args[1]),
                                   stmts, curr_token_id);

        curr_stmt_with_token_id(call, stmts, curr_token_id);

        process_broadcast_barrier(call, curr_token_id, curr_barrier_id, stmts);

        return SeqStmt::Flatten(stmts);
      }
    }

    Array<Stmt> stmts;
    token_process_prim_expr(op->value, stmts);
    stmts.push_back(StmtMutator::VisitStmt_(op));
    return SeqStmt::Flatten(stmts);
  }

  // Handles control flow splitting (IfThenElse).
  // We need to track buffer states independently for then/else branches and
  // then merge them.
  Stmt VisitStmt_(const IfThenElseNode *op) {
    Array<Stmt> stmts;
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
    token_process_prim_expr(loop->min, stmts);
    token_process_prim_expr(loop->extent, stmts);

    LoopAsyncCollector collector;
    collector(loop->body);

    LoopScope scope;
    scope.loop_var = loop->loop_var;
    scope.loop_min = loop->min;
    scope.loop_extent = loop->extent;
    scope.async_ops = collector.async_ops;
    for (auto &async_op : scope.async_ops) {
      int token = GetNextTokenId();
      async_op.token = token;
      pre_assigned_tokens_[async_op.op] = token;

      const CallNode *call = async_op.call;
      scope.token_to_call[token] = call;

      // check if it is a broadcast
      if (call && call->op.same_as(broadcast_())) {
        int barrier = GetNextBarrierId();
        token_to_barrier_map.Set(token, barrier);
        barrier_to_token_map.Set(barrier, token);
      }
    }

    AnalyzeLoopCarriedDependencies(&scope);

    loop_scopes_.push_back(scope);

    Stmt loop_stmt = StmtMutator::VisitStmt_(loop);

    scope = loop_scopes_.back();
    loop_scopes_.pop_back();
    for (const auto &async_op : scope.async_ops) {
      pre_assigned_tokens_.erase(async_op.op);
    }

    InjectLoopEntryNullTokens(scope, stmts);
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
  int GetNextBarrierId() { return barrier_count++; }

  int token_count;
  int barrier_count;
  int mesh_nrow_;
  int mesh_ncol_;
  arith::Analyzer *analyzer_;

  Array<Array<ObjectRef>> read_buffers;
  Array<Array<ObjectRef>> write_buffers;
  Map<Array<ObjectRef>, int> read_buffer_token_map;
  Map<Array<ObjectRef>, int> write_buffer_token_map;
  Map<int, int> token_to_barrier_map;
  Map<int, int> barrier_to_token_map;

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<LoopScope> loop_scopes_;
  std::map<const EvaluateNode *, int> pre_assigned_tokens_;
};

// Rewriter to inject final synchronization waits before the device function
// returns. This ensures all pending asynchronous operations are completed
// before the device kernel finishes, handling both explicit returns and
// implicit function exits.
class DeviceFuncWaitRewriter : public StmtMutator {
public:
  DeviceFuncWaitRewriter(Map<int, int> token_to_barrier_map)
      : token_to_barrier_map_(std::move(token_to_barrier_map)) {}

  Stmt operator()(Stmt body) { return this->VisitStmt(body); }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      Stmt body = StmtMutator::VisitStmt(op->body);

      DeviceTokenCollector collector;
      collector(body);

      if (collector.tokens.empty()) {
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }

      Array<Stmt> stmts;
      if (const auto *seq = body.as<SeqStmtNode>()) {
        stmts = seq->seq;
      } else {
        stmts.push_back(body);
      }

      std::vector<int> tokens(collector.tokens.begin(), collector.tokens.end());
      std::sort(tokens.begin(), tokens.end());

      for (int token_id : tokens) {
        stmts.push_back(Evaluate(Call(DataType::Handle(), wait_token(),
                                      {IntImm(DataType::Int(32), token_id)})));
        if (token_to_barrier_map_.count(token_id)) {
          int barrier_id = token_to_barrier_map_[token_id];
          stmts.push_back(
              Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                            {IntImm(DataType::Int(32), barrier_id)})));
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
  Map<int, int> token_to_barrier_map_;

  // Helper to collect all token IDs referenced within the device block.
  class DeviceTokenCollector : public StmtExprVisitor {
  public:
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(sync_token_id())) {
        int token_id = op->args[0].as<IntImm>().value()->value;
        tokens.insert(token_id);
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    std::set<int> tokens;
  };
};

// Collector to identify all sync tokens and barriers generated within a given
// statement or expression. This is primarily used for tracking resources that
// may need subsequent synchronizations.
class AsyncResourceCollector : public StmtExprVisitor {
public:
  std::set<int> generated_tokens;
  std::set<int> generated_barriers;

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(sync_token_id()) || op->op.same_as(sync_null_token())) {
      if (!op->args.empty() && op->args[0].as<IntImmNode>()) {
        int token_id = op->args[0].as<IntImmNode>()->value;
        generated_tokens.insert(token_id);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(barrier_init())) {
        if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
          int barrier_id = call->args[0].as<IntImmNode>()->value;
          generated_barriers.insert(barrier_id);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

// Analyzer to track which tokens and barriers are currently pending (i.e.,
// generated but not yet waited on) within a specific execution scope. Used to
// determine if additional waits are required. Particularly note the following
// scenario: dependent tokens (or barriers) within a loop may lack a
// corresponding wait (or arrive_and_wait) after the final iteration.
class PendingAnalyzer : public StmtExprVisitor {
public:
  PendingAnalyzer(Map<int, int> barrier_to_token_map)
      : barrier_to_token_map_(barrier_to_token_map) {}

  std::set<int> pending_tokens;
  std::set<int> pending_barriers;
  Map<int, int> barrier_to_token_map_;

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
      } else if (call->op.same_as(barrier_init())) {
        if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
          pending_barriers.insert(call->args[0].as<IntImmNode>()->value);
        }
      } else if (call->op.same_as(barrier_arrive_and_wait())) {
        if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
          int barrier_id = call->args[0].as<IntImmNode>()->value;
          pending_barriers.erase(barrier_id);
          if (barrier_to_token_map_.count(barrier_id)) {
            pending_tokens.erase(barrier_to_token_map_[barrier_id]);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    auto pending_tokens_before = pending_tokens;
    auto pending_barriers_before = pending_barriers;

    VisitStmt(op->then_case);
    auto then_pending_tokens = pending_tokens;
    auto then_pending_barriers = pending_barriers;

    pending_tokens = pending_tokens_before;
    pending_barriers = pending_barriers_before;

    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }

    pending_tokens.insert(then_pending_tokens.begin(),
                          then_pending_tokens.end());
    pending_barriers.insert(then_pending_barriers.begin(),
                            then_pending_barriers.end());
  }

  void VisitStmt_(const ForNode *op) final { VisitStmt(op->body); }

  void VisitStmt_(const WhileNode *op) final { VisitStmt(op->body); }

  void VisitStmt_(const SeqStmtNode *op) final {
    for (auto stmt : op->seq) {
      VisitStmt(stmt);
    }
  }
};

// Collector to identify all sync tokens and barriers that are explicitly waited
// on within a given statement.
class ResolvedResourceCollector : public StmtExprVisitor {
public:
  std::set<int> resolved_tokens;
  std::set<int> resolved_barriers;
  Map<int, int> barrier_to_token_map_;

  ResolvedResourceCollector(Map<int, int> barrier_to_token_map)
      : barrier_to_token_map_(std::move(barrier_to_token_map)) {}

  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(wait_token())) {
        if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
          resolved_tokens.insert(call->args[0].as<IntImmNode>()->value);
        }
      } else if (call->op.same_as(barrier_arrive_and_wait())) {
        if (!call->args.empty() && call->args[0].as<IntImmNode>()) {
          int barrier_id = call->args[0].as<IntImmNode>()->value;
          resolved_barriers.insert(barrier_id);
          if (barrier_to_token_map_.count(barrier_id)) {
            resolved_tokens.insert(barrier_to_token_map_[barrier_id]);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

// Optimization pass to remove redundant synchronization calls.
// If a token or barrier has already been waited on in the current execution
// path, subsequent waits are unnecessary.
class EliminateRedundancyRewriter : public StmtMutator {
public:
  EliminateRedundancyRewriter(arith::Analyzer *analyzer = nullptr,
                              std::vector<int> parent_token_ids = {},
                              std::vector<int> parent_barrier_ids = {},
                              Map<int, int> barrier_to_token_map = {})
      : analyzer_(analyzer), parent_token_ids_(std::move(parent_token_ids)),
        parent_barrier_ids_(std::move(parent_barrier_ids)),
        barrier_to_token_map_(std::move(barrier_to_token_map)) {
    current_token_ids_ = {};
    current_barrier_ids_ = {};
  }

  std::vector<int> get_current_barrier_ids() const {
    return current_barrier_ids_;
  }

  std::vector<int> get_current_token_ids() const { return current_token_ids_; }

private:
  std::vector<int> get_all_token_ids() const {
    std::vector<int> all_token_ids = parent_token_ids_;
    all_token_ids.insert(all_token_ids.end(), current_token_ids_.begin(),
                         current_token_ids_.end());
    return all_token_ids;
  }

  std::vector<int> get_all_barrier_ids() const {
    std::vector<int> all_barrier_ids = parent_barrier_ids_;
    all_barrier_ids.insert(all_barrier_ids.end(), current_barrier_ids_.begin(),
                           current_barrier_ids_.end());
    return all_barrier_ids;
  }

  // Propagates the resolved token and barrier states from a block (e.g., loop
  // body or if branch) to the current scope, marking them as handled to avoid
  // redundant waits.
  void PropagateResolvedStates(const Stmt &block,
                               bool guaranteed_to_execute = false) {
    // Collect async resources that are created inside this block. These IDs
    // represent potential synchronization points introduced by the rewriter
    // (e.g., new tokens and barriers).
    AsyncResourceCollector collector;
    collector(block);

    // Analyze which of the collected resources are still pending after the
    // block finishes. A resource is considered "pending" if there exists a
    // path in the block that may still require a corresponding wait later.
    PendingAnalyzer pending_analyzer(barrier_to_token_map_);
    pending_analyzer(block);

    // If a barrier is generated in this block but is not pending at the block
    // exit, it means the block has fully synchronized that barrier internally
    // (or it has no remaining uses). We can mark it as resolved in the current
    // scope so parent scopes won't emit redundant waits for it.
    for (int barrier_id : collector.generated_barriers) {
      if (pending_analyzer.pending_barriers.count(barrier_id) == 0) {
        if (std::find(current_barrier_ids_.begin(), current_barrier_ids_.end(),
                      barrier_id) == current_barrier_ids_.end()) {
          current_barrier_ids_.push_back(barrier_id);
        }
      }
    }

    // Same propagation for tokens: if a token is generated in the block and is
    // not pending at the block exit, then any necessary waits for that token
    // have been handled within the block. Record it as resolved for the
    // current scope to avoid re-waiting in enclosing control-flow.
    for (int token_id : collector.generated_tokens) {
      if (pending_analyzer.pending_tokens.count(token_id) == 0) {
        if (std::find(current_token_ids_.begin(), current_token_ids_.end(),
                      token_id) == current_token_ids_.end()) {
          current_token_ids_.push_back(token_id);
        }
      }
    }

    // If the block is guaranteed to execute, any explicit waits within the
    // block that are not pending at the end are also resolved for the parent
    // scope.
    if (guaranteed_to_execute) {
      ResolvedResourceCollector resolved_collector(barrier_to_token_map_);
      resolved_collector(block);

      for (int barrier_id : resolved_collector.resolved_barriers) {
        if (pending_analyzer.pending_barriers.count(barrier_id) == 0) {
          if (std::find(current_barrier_ids_.begin(),
                        current_barrier_ids_.end(),
                        barrier_id) == current_barrier_ids_.end()) {
            current_barrier_ids_.push_back(barrier_id);
          }
        }
      }

      for (int token_id : resolved_collector.resolved_tokens) {
        if (pending_analyzer.pending_tokens.count(token_id) == 0) {
          if (std::find(current_token_ids_.begin(), current_token_ids_.end(),
                        token_id) == current_token_ids_.end()) {
            current_token_ids_.push_back(token_id);
          }
        }
      }
    }
  }

  // Intercepts wait_token and barrier_arrive_and_wait calls.
  // Drops the statement if the synchronization has already been performed in
  // the current execution path.
  Stmt VisitStmt_(const EvaluateNode *op) {
    const CallNode *call = op->value.as<CallNode>();
    if (call) {
      if (call->op.same_as(wait_token())) {
        int token_id = call->args[0].as<IntImm>().value()->value;
        // if the token_id is in parent_token_ids or current_token_ids, it means
        // the wait is redundant and can be eliminated
        if (std::find(parent_token_ids_.begin(), parent_token_ids_.end(),
                      token_id) != parent_token_ids_.end() ||
            std::find(current_token_ids_.begin(), current_token_ids_.end(),
                      token_id) != current_token_ids_.end()) {
          // eliminate this wait and do not add it to stmts
          return Stmt();
        } else {
          current_token_ids_.push_back(token_id);
          return StmtMutator::VisitStmt_(op);
        }
      } else if (call->op.same_as(barrier_arrive_and_wait())) {
        int barrier_id = call->args[0].as<IntImm>().value()->value;
        // if the barrier_id is in parent_barrier_ids or current_barrier_ids, it
        // means the barrier wait is redundant and can be eliminated
        if (std::find(parent_barrier_ids_.begin(), parent_barrier_ids_.end(),
                      barrier_id) != parent_barrier_ids_.end() ||
            std::find(current_barrier_ids_.begin(), current_barrier_ids_.end(),
                      barrier_id) != current_barrier_ids_.end()) {
          // eliminate this barrier wait and do not add it to stmts
          return Stmt();
        } else {
          current_barrier_ids_.push_back(barrier_id);
          if (barrier_to_token_map_.count(barrier_id)) {
            current_token_ids_.push_back(barrier_to_token_map_[barrier_id]);
          }
          return StmtMutator::VisitStmt_(op);
        }
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) {
    auto eliminate_sync_then_rewriter = EliminateRedundancyRewriter(
        analyzer_, get_all_token_ids(), get_all_barrier_ids(),
        barrier_to_token_map_);
    auto then_case = eliminate_sync_then_rewriter(op->then_case);

    Stmt else_case;
    if (op->else_case.defined()) {
      auto eliminate_sync_else_rewriter = EliminateRedundancyRewriter(
          analyzer_, get_all_token_ids(), get_all_barrier_ids(),
          barrier_to_token_map_);
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

      std::vector<int> then_barriers =
          eliminate_sync_then_rewriter.get_current_barrier_ids();
      std::vector<int> else_barriers =
          eliminate_sync_else_rewriter.get_current_barrier_ids();
      for (int b_id : then_barriers) {
        if (std::find(else_barriers.begin(), else_barriers.end(), b_id) !=
            else_barriers.end()) {
          if (std::find(current_barrier_ids_.begin(),
                        current_barrier_ids_.end(),
                        b_id) == current_barrier_ids_.end()) {
            current_barrier_ids_.push_back(b_id);
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
        analyzer_, get_all_token_ids(), get_all_barrier_ids(),
        barrier_to_token_map_);
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
        analyzer_, get_all_token_ids(), get_all_barrier_ids(),
        barrier_to_token_map_);
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
  // Barrier IDs that are already known to be arrived-and-waited in outer scopes
  std::vector<int> parent_barrier_ids_;
  // Barrier IDs that have been arrived-and-waited along the current execution
  // path
  std::vector<int> current_barrier_ids_;
  Map<int, int> barrier_to_token_map_;
};

class HoistLoopWaitRewriter : public StmtMutator {
public:
  Stmt operator()(Stmt body) { return VisitStmt(body); }

private:
  struct WaitAction {
    enum class Kind { kToken, kBarrier };
    Kind kind;
    int id;
  };

  static bool MatchWaitTokenStmt(const Stmt &s, int *token_id) {
    const auto *eval = s.as<EvaluateNode>();
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

  static bool MatchBarrierWaitStmt(const Stmt &s, int *barrier_id) {
    const auto *eval = s.as<EvaluateNode>();
    if (!eval) {
      return false;
    }
    const auto *call = eval->value.as<CallNode>();
    if (!call || !call->op.same_as(barrier_arrive_and_wait()) ||
        call->args.size() != 1) {
      return false;
    }
    const auto *imm = call->args[0].as<IntImmNode>();
    if (!imm) {
      return false;
    }
    *barrier_id = imm->value;
    return true;
  }

  static Stmt MakeWaitTokenStmt(int token_id) {
    return Evaluate(Call(DataType::Handle(), wait_token(),
                         {IntImm(DataType::Int(32), token_id)}));
  }

  static Stmt MakeBarrierWaitStmt(int barrier_id) {
    return Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                         {IntImm(DataType::Int(32), barrier_id)}));
  }

  class RemoveWaitsRewriter : public StmtMutator {
  public:
    RemoveWaitsRewriter(std::unordered_set<int> tokens,
                        std::unordered_set<int> barriers)
        : tokens_(std::move(tokens)), barriers_(std::move(barriers)) {}

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> out;
      out.reserve(op->seq.size());
      for (const Stmt &s : op->seq) {
        Stmt ns = VisitStmt(s);
        if (ns.defined()) {
          out.push_back(ns);
        }
      }
      return SeqStmt::Flatten(out);
    }

    Stmt VisitStmt_(const EvaluateNode *op) final {
      const CallNode *call = op->value.as<CallNode>();
      if (!call) {
        return StmtMutator::VisitStmt_(op);
      }
      if (call->op.same_as(wait_token()) && call->args.size() == 1) {
        if (const auto *imm = call->args[0].as<IntImmNode>()) {
          if (tokens_.count(imm->value)) {
            return Stmt();
          }
        }
      }
      if (call->op.same_as(barrier_arrive_and_wait()) &&
          call->args.size() == 1) {
        if (const auto *imm = call->args[0].as<IntImmNode>()) {
          if (barriers_.count(imm->value)) {
            return Stmt();
          }
        }
      }
      return StmtMutator::VisitStmt_(op);
    }

  private:
    std::unordered_set<int> tokens_;
    std::unordered_set<int> barriers_;
  };

  struct HoistPlan {
    std::vector<WaitAction> actions;
    std::unordered_set<int> tokens_to_remove;
    std::unordered_set<int> barriers_to_remove;
  };

  class LoopWaitCollector : public StmtVisitor {
  public:
    LoopWaitCollector(const std::set<int> &available_tokens,
                      const std::set<int> &available_barriers,
                      const std::set<int> &generated_tokens_in_loop,
                      const std::set<int> &generated_barriers_in_loop)
        : available_tokens_(available_tokens),
          available_barriers_(available_barriers),
          generated_tokens_in_loop_(generated_tokens_in_loop),
          generated_barriers_in_loop_(generated_barriers_in_loop) {}

    HoistPlan plan;

    void VisitStmt_(const SeqStmtNode *op) final {
      int n = static_cast<int>(op->seq.size());
      int i = 0;
      while (i < n) {
        int t = -1;
        int b = -1;
        if (i + 1 < n && MatchWaitTokenStmt(op->seq[i], &t) &&
            MatchBarrierWaitStmt(op->seq[i + 1], &b)) {
          bool token_ok =
              available_tokens_.count(t) && !generated_tokens_in_loop_.count(t);
          bool barrier_ok = available_barriers_.count(b) &&
                            !generated_barriers_in_loop_.count(b);
          if (token_ok && barrier_ok) {
            plan.actions.push_back({WaitAction::Kind::kToken, t});
            plan.actions.push_back({WaitAction::Kind::kBarrier, b});
            plan.tokens_to_remove.insert(t);
            plan.barriers_to_remove.insert(b);
          }
          i += 2;
          continue;
        }

        if (MatchWaitTokenStmt(op->seq[i], &t)) {
          bool token_ok =
              available_tokens_.count(t) && !generated_tokens_in_loop_.count(t);
          if (token_ok) {
            plan.actions.push_back({WaitAction::Kind::kToken, t});
            plan.tokens_to_remove.insert(t);
          }
          i += 1;
          continue;
        }

        if (MatchBarrierWaitStmt(op->seq[i], &b)) {
          bool barrier_ok = available_barriers_.count(b) &&
                            !generated_barriers_in_loop_.count(b);
          if (barrier_ok) {
            plan.actions.push_back({WaitAction::Kind::kBarrier, b});
            plan.barriers_to_remove.insert(b);
          }
          i += 1;
          continue;
        }

        VisitStmt(op->seq[i]);
        i += 1;
      }
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      VisitStmt(op->then_case);
      if (op->else_case.defined()) {
        VisitStmt(op->else_case.value());
      }
    }

    void VisitStmt_(const ForNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const WhileNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const AttrStmtNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const LetStmtNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const AllocateNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const AssertStmtNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const BufferRealizeNode *op) final { VisitStmt(op->body); }

    void VisitStmt_(const BlockRealizeNode *op) final { VisitStmt(op->block); }

    void VisitStmt_(const BlockNode *op) final { VisitStmt(op->body); }

  private:
    const std::set<int> &available_tokens_;
    const std::set<int> &available_barriers_;
    const std::set<int> &generated_tokens_in_loop_;
    const std::set<int> &generated_barriers_in_loop_;
  };

  struct HoistResult {
    Array<Stmt> hoisted;
    Stmt loop_stmt;
  };

  static void UpdateAvailability(const Stmt &s, std::set<int> *available_tokens,
                                 std::set<int> *available_barriers) {
    AsyncResourceCollector collector;
    collector(s);
    for (int t : collector.generated_tokens) {
      available_tokens->insert(t);
    }
    for (int b : collector.generated_barriers) {
      available_barriers->insert(b);
    }
  }

  HoistResult HoistFromFor(const ForNode *op,
                           const std::set<int> &available_tokens,
                           const std::set<int> &available_barriers) {
    AsyncResourceCollector loop_resources;
    loop_resources(op->body);

    LoopWaitCollector collector(available_tokens, available_barriers,
                                loop_resources.generated_tokens,
                                loop_resources.generated_barriers);
    collector(op->body);

    HoistPlan plan = std::move(collector.plan);
    if (plan.actions.empty()) {
      return {Array<Stmt>(), ffi::GetRef<Stmt>(op)};
    }

    std::unordered_set<int> emitted_tokens;
    std::unordered_set<int> emitted_barriers;
    Array<Stmt> hoisted;
    for (const auto &action : plan.actions) {
      if (action.kind == WaitAction::Kind::kToken) {
        if (!emitted_tokens.count(action.id)) {
          hoisted.push_back(MakeWaitTokenStmt(action.id));
          emitted_tokens.insert(action.id);
        }
      } else {
        if (!emitted_barriers.count(action.id)) {
          hoisted.push_back(MakeBarrierWaitStmt(action.id));
          emitted_barriers.insert(action.id);
        }
      }
    }

    RemoveWaitsRewriter remover(std::move(plan.tokens_to_remove),
                                std::move(plan.barriers_to_remove));
    Stmt new_body = remover(op->body);
    Stmt new_loop = For(op->loop_var, op->min, op->extent, op->kind, new_body,
                        op->thread_binding, op->annotations);
    return {hoisted, new_loop};
  }

  HoistResult HoistFromWhile(const WhileNode *op,
                             const std::set<int> &available_tokens,
                             const std::set<int> &available_barriers) {
    AsyncResourceCollector loop_resources;
    loop_resources(op->body);

    LoopWaitCollector collector(available_tokens, available_barriers,
                                loop_resources.generated_tokens,
                                loop_resources.generated_barriers);
    collector(op->body);

    HoistPlan plan = std::move(collector.plan);
    if (plan.actions.empty()) {
      return {Array<Stmt>(), ffi::GetRef<Stmt>(op)};
    }

    std::unordered_set<int> emitted_tokens;
    std::unordered_set<int> emitted_barriers;
    Array<Stmt> hoisted;
    for (const auto &action : plan.actions) {
      if (action.kind == WaitAction::Kind::kToken) {
        if (!emitted_tokens.count(action.id)) {
          hoisted.push_back(MakeWaitTokenStmt(action.id));
          emitted_tokens.insert(action.id);
        }
      } else {
        if (!emitted_barriers.count(action.id)) {
          hoisted.push_back(MakeBarrierWaitStmt(action.id));
          emitted_barriers.insert(action.id);
        }
      }
    }

    RemoveWaitsRewriter remover(std::move(plan.tokens_to_remove),
                                std::move(plan.barriers_to_remove));
    Stmt new_body = remover(op->body);
    Stmt new_loop = While(op->condition, new_body);
    return {hoisted, new_loop};
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    std::set<int> available_tokens;
    std::set<int> available_barriers;

    Array<Stmt> out;
    out.reserve(op->seq.size());
    for (const Stmt &s : op->seq) {
      available_tokens_ = available_tokens;
      available_barriers_ = available_barriers;

      Stmt ns = VisitStmt(s);
      if (ns.defined()) {
        out.push_back(ns);
        UpdateAvailability(ns, &available_tokens, &available_barriers);
      }
    }

    available_tokens_ = available_tokens;
    available_barriers_ = available_barriers;
    return SeqStmt::Flatten(out);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto entry_tokens = available_tokens_;
    auto entry_barriers = available_barriers_;

    available_tokens_ = entry_tokens;
    available_barriers_ = entry_barriers;
    Stmt then_case = VisitStmt(op->then_case);
    auto then_end_tokens = available_tokens_;
    auto then_end_barriers = available_barriers_;

    ffi::Optional<Stmt> else_case = std::nullopt;
    auto else_end_tokens = entry_tokens;
    auto else_end_barriers = entry_barriers;
    if (op->else_case.defined()) {
      available_tokens_ = entry_tokens;
      available_barriers_ = entry_barriers;
      Stmt else_stmt = VisitStmt(op->else_case.value());
      else_case = else_stmt;
      else_end_tokens = available_tokens_;
      else_end_barriers = available_barriers_;
    }

    available_tokens_ = entry_tokens;
    available_barriers_ = entry_barriers;
    UpdateAvailability(then_case, &available_tokens_, &available_barriers_);
    if (else_case.defined()) {
      UpdateAvailability(else_case.value(), &available_tokens_,
                         &available_barriers_);
    }

    for (int t : then_end_tokens) {
      available_tokens_.insert(t);
    }
    for (int t : else_end_tokens) {
      available_tokens_.insert(t);
    }
    for (int b : then_end_barriers) {
      available_barriers_.insert(b);
    }
    for (int b : else_end_barriers) {
      available_barriers_.insert(b);
    }

    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto entry_tokens = available_tokens_;
    auto entry_barriers = available_barriers_;

    PrimExpr min = VisitExpr(op->min);
    PrimExpr extent = VisitExpr(op->extent);
    Stmt new_body = VisitStmt(op->body);
    available_tokens_ = entry_tokens;
    available_barriers_ = entry_barriers;
    Stmt loop = For(op->loop_var, min, extent, op->kind, new_body,
                    op->thread_binding, op->annotations);

    HoistResult res = HoistFromFor(loop.as<ForNode>(), available_tokens_,
                                   available_barriers_);
    if (res.hoisted.empty()) {
      UpdateAvailability(loop, &available_tokens_, &available_barriers_);
      return loop;
    }
    Array<Stmt> seq = res.hoisted;
    seq.push_back(res.loop_stmt);
    Stmt out = SeqStmt::Flatten(seq);
    UpdateAvailability(out, &available_tokens_, &available_barriers_);
    return out;
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    auto entry_tokens = available_tokens_;
    auto entry_barriers = available_barriers_;

    PrimExpr condition = VisitExpr(op->condition);
    Stmt new_body = VisitStmt(op->body);
    available_tokens_ = entry_tokens;
    available_barriers_ = entry_barriers;
    Stmt loop = While(condition, new_body);

    HoistResult res = HoistFromWhile(loop.as<WhileNode>(), available_tokens_,
                                     available_barriers_);
    if (res.hoisted.empty()) {
      UpdateAvailability(loop, &available_tokens_, &available_barriers_);
      return loop;
    }
    Array<Stmt> seq = res.hoisted;
    seq.push_back(res.loop_stmt);
    Stmt out = SeqStmt::Flatten(seq);
    UpdateAvailability(out, &available_tokens_, &available_barriers_);
    return out;
  }

private:
  std::set<int> available_tokens_;
  std::set<int> available_barriers_;
};

class CompactSyncIdsRewriter : public StmtExprMutator {
public:
  Stmt operator()(Stmt body) {
    SyncIdCollector collector;
    collector(body);
    token_id_map_ = BuildDenseMap(collector.token_ids);
    barrier_id_map_ = BuildDenseMap(collector.barrier_ids);
    return VisitStmt(body);
  }

private:
  class SyncIdCollector : public StmtExprVisitor {
  public:
    void VisitExpr_(const CallNode *op) final {
      if (IsTokenCall(op)) {
        CollectId(op, &token_ids);
      } else if (IsBarrierCall(op)) {
        CollectId(op, &barrier_ids);
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    std::set<int> token_ids;
    std::set<int> barrier_ids;

  private:
    static bool IsTokenCall(const CallNode *op) {
      return op->op.same_as(sync_token_id()) ||
             op->op.same_as(sync_null_token()) || op->op.same_as(wait_token());
    }

    static bool IsBarrierCall(const CallNode *op) {
      return op->op.same_as(barrier_init()) ||
             op->op.same_as(barrier_arrive_and_wait());
    }

    static void CollectId(const CallNode *op, std::set<int> *ids) {
      if (op->args.empty()) {
        return;
      }
      if (const auto *imm = op->args[0].as<IntImmNode>()) {
        ids->insert(imm->value);
      }
    }
  };

  static std::unordered_map<int, int> BuildDenseMap(const std::set<int> &ids) {
    std::unordered_map<int, int> id_map;
    int next_id = 0;
    for (int old_id : ids) {
      id_map.emplace(old_id, next_id++);
    }
    return id_map;
  }

  static bool IsTokenCall(const CallNode *op) {
    return op->op.same_as(sync_token_id()) ||
           op->op.same_as(sync_null_token()) || op->op.same_as(wait_token());
  }

  static bool IsBarrierCall(const CallNode *op) {
    return op->op.same_as(barrier_init()) ||
           op->op.same_as(barrier_arrive_and_wait());
  }

  PrimExpr RemapFirstArg(const CallNode *op,
                         const std::unordered_map<int, int> &id_map) {
    if (op->args.empty()) {
      return ffi::GetRef<PrimExpr>(op);
    }

    const auto *imm = op->args[0].as<IntImmNode>();
    if (!imm) {
      return ffi::GetRef<PrimExpr>(op);
    }

    auto it = id_map.find(imm->value);
    if (it == id_map.end()) {
      return ffi::GetRef<PrimExpr>(op);
    }

    Array<PrimExpr> new_args = op->args;
    new_args.Set(0, IntImm(imm->dtype, it->second));
    return Call(op->dtype, op->op, new_args, op->annotations, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *call = expr.as<CallNode>();
    if (!call) {
      return expr;
    }
    if (IsTokenCall(call)) {
      return RemapFirstArg(call, token_id_map_);
    }
    if (IsBarrierCall(call)) {
      return RemapFirstArg(call, barrier_id_map_);
    }
    return expr;
  }

  std::unordered_map<int, int> token_id_map_;
  std::unordered_map<int, int> barrier_id_map_;
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
    SunmmioMeshConfig mesh = GetSunmmioMeshConfig(target);
    int mesh_nrow = mesh.nrow;
    int mesh_ncol = mesh.ncol;

    auto inject_sync_rewriter =
        InjectSyncRewriter(f->buffer_map, mesh_nrow, mesh_ncol, analyzer);
    f.CopyOnWrite()->body = inject_sync_rewriter(f->body);

    auto device_func_wait_rewriter =
        DeviceFuncWaitRewriter(inject_sync_rewriter.get_token_to_barrier_map());
    f.CopyOnWrite()->body = device_func_wait_rewriter(f->body);

    auto hoist_loop_wait_rewriter = HoistLoopWaitRewriter();
    f.CopyOnWrite()->body = hoist_loop_wait_rewriter(f->body);

    auto eliminate_redundancy_rewriter = EliminateRedundancyRewriter(
        analyzer, std::vector<int>({}), std::vector<int>({}),
        inject_sync_rewriter.get_barrier_to_token_map());
    f.CopyOnWrite()->body = eliminate_redundancy_rewriter(f->body);

    auto compact_sync_ids_rewriter = CompactSyncIdsRewriter();
    f.CopyOnWrite()->body = compact_sync_ids_rewriter(f->body);

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
