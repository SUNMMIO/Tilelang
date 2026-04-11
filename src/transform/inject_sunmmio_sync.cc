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

#include <set>
#include <utility>

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
class LoopAsyncCollector : public StmtVisitor {
public:
  void VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = op->value.as<CallNode>();
    if (call) {
      if (call->op.same_as(dma_copy())) {
        reads.push_back({op, NormalizeToBufferRegion(call->args[0])});
        writes.push_back({op, NormalizeToBufferRegion(call->args[1])});
        ops.push_back(op);
      } else if (call->op.same_as(mma_sunmmio())) {
        reads.push_back({op, NormalizeToBufferRegion(call->args[0])});
        reads.push_back({op, NormalizeToBufferRegion(call->args[1])});
        writes.push_back({op, NormalizeToBufferRegion(call->args[2])});
        ops.push_back(op);
      } else if (call->op.same_as(broadcast_())) {
        reads.push_back({op, NormalizeToBufferRegion(call->args[0])});
        writes.push_back({op, NormalizeToBufferRegion(call->args[1])});
        ops.push_back(op);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  std::vector<std::pair<const EvaluateNode *, BufferRegion>> reads;
  std::vector<std::pair<const EvaluateNode *, BufferRegion>> writes;
  std::vector<const EvaluateNode *> ops;
};

// Represents the scope of a loop for dependency tracking.
// Stores writes that happen within the loop to check for loop-carried
// dependencies.
struct LoopScope {
  Array<Array<ObjectRef>> writes;
  Array<Array<ObjectRef>> reads;
  Map<Array<ObjectRef>, int> buffer_ref_to_token;
  std::map<int, const CallNode *> token_to_call;
  // When memory accesses within the loop body depend on asynchronous operations
  // from previous iterations (which are identified by pre-assigned tokens), a
  // corresponding sync_null_token must be inserted before the loop because
  // those asynchronous operations have not yet occurred during the first
  // iteration. The waited_tokens set is used to record these tokens.
  std::set<int> waited_tokens;
};

// Main rewriter class to inject synchronization primitives.
// It tracks buffer accesses and inserts wait_token and barrier_wait calls
// to enforce correct ordering based on data dependencies.
class InjectSyncRewriter : public StmtMutator {
public:
  InjectSyncRewriter(Map<Var, Buffer> buffer_data_to_buffer, int mesh_nrow,
                     int mesh_ncol)
      : buffer_data_to_buffer_(buffer_data_to_buffer), mesh_nrow_(mesh_nrow),
        mesh_ncol_(mesh_ncol) {
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

  // Analyzes a read operation on a buffer region.
  // Checks for dependencies with pending writes (RAW) and inserts waits if
  // necessary. Records the read access for future dependency checks.
  void token_process_read_buffer(const BufferRegion &buffer_region,
                                 Array<Stmt> &stmts, int curr_token_id,
                                 bool is_log_async = true) {
    Buffer src_buffer = buffer_region->buffer;
    Region src_region = buffer_region->region;
    auto src = Array<ObjectRef>{src_buffer, src_region};
    // Tracks whether a token has already been waited on within the current loop
    // level or in any of the scopes recorded in loop_scopes .
    std::unordered_set<int> waited_tokens;

    // Check if the current read buffer has dependencies with existing write
    // buffers. If yes, we need to wait for the write to finish before reading.
    for (const Array<ObjectRef> &buf : write_buffers) {
      if (is_log_async && write_buffer_token_map[buf] == curr_token_id) {
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

    // Check loop carried dependencies
    for (int i = loop_scopes_.size() - 1; i >= 0; i--) {
      auto &scope = loop_scopes_[i];
      for (const auto &buf : scope.writes) {
        if (is_log_async && scope.buffer_ref_to_token[buf] == curr_token_id) {
          continue;
        }
        Buffer buf_buffer = Downcast<Buffer>(buf[0]);
        Region buf_region = Downcast<Region>(buf[1]);
        if (src_buffer.same_as(buf_buffer) &&
            RegionIntersect(src_region, buf_region)) {
          int token = scope.buffer_ref_to_token[buf];
          if (waited_tokens.count(token) == 0) {
            process_wait_token_and_barrier_wait(stmts, token);
            waited_tokens.insert(token);
            scope.waited_tokens.insert(token);
          }
        }
      }
    }

    // After processing the dependencies with existing buffers, we can add the
    // current read buffer to the list.
    if (is_log_async) {
      read_buffers.push_back(src);
      read_buffer_token_map.Set(src, curr_token_id);
    }
  }

  // Analyzes a write operation on a buffer region.
  // Checks for dependencies with pending reads (WAR) and writes (WAW).
  // Inserts waits if necessary and records the write access.
  void token_process_write_buffer(const BufferRegion &buffer_region,
                                  Array<Stmt> &stmts, int curr_token_id,
                                  bool is_log_async = true) {
    Buffer dst_buffer = buffer_region->buffer;
    Region dst_region = buffer_region->region;
    auto dst = Array<ObjectRef>{dst_buffer, dst_region};
    std::unordered_set<int> waited_tokens;

    // Check if the current write buffer has dependencies with existing read
    // buffers. If yes, we need to wait for the read to finish before writing.
    for (const Array<ObjectRef> &buf : read_buffers) {
      if (is_log_async && read_buffer_token_map[buf] == curr_token_id) {
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
      if (is_log_async && write_buffer_token_map[buf] == curr_token_id) {
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

    // Check loop carried dependencies
    for (int i = loop_scopes_.size() - 1; i >= 0; i--) {
      auto &scope = loop_scopes_[i];
      for (const auto &buf : scope.writes) {
        if (is_log_async && scope.buffer_ref_to_token[buf] == curr_token_id) {
          continue;
        }
        Buffer buf_buffer = Downcast<Buffer>(buf[0]);
        Region buf_region = Downcast<Region>(buf[1]);
        if (dst_buffer.same_as(buf_buffer) &&
            RegionIntersect(dst_region, buf_region)) {
          int token = scope.buffer_ref_to_token[buf];
          if (waited_tokens.count(token) == 0) {
            process_wait_token_and_barrier_wait(stmts, token);
            waited_tokens.insert(token);
            scope.waited_tokens.insert(token);
          }
        }
      }
      for (const auto &buf : scope.reads) {
        if (is_log_async && scope.buffer_ref_to_token[buf] == curr_token_id) {
          continue;
        }
        Buffer buf_buffer = Downcast<Buffer>(buf[0]);
        Region buf_region = Downcast<Region>(buf[1]);
        if (dst_buffer.same_as(buf_buffer) &&
            RegionIntersect(dst_region, buf_region)) {
          int token = scope.buffer_ref_to_token[buf];
          if (waited_tokens.count(token) == 0) {
            process_wait_token_and_barrier_wait(stmts, token);
            waited_tokens.insert(token);
            scope.waited_tokens.insert(token);
          }
        }
      }
    }

    // After processing the dependencies with existing buffers, we can add the
    // current write buffer to the list.
    if (is_log_async) {
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
                     Integer read_core, Array<Integer> write_cores = {}) {
    Array<PrimExpr> args;
    args.push_back(barrier_id);
    args.push_back(read_core);
    if (!write_cores.empty()) {
      for (const auto &core : write_cores) {
        if (core->value != read_core->value) {
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
    auto src_core = call->args[3].as<Integer>().value();
    int direction = call->args[4].as<IntImm>().value()->value;
    Array<int> masks;
    for (size_t i = 5; i < call->args.size(); i++) {
      masks.push_back(call->args[i].as<IntImm>().value()->value);
    }

    int src_core_row = src_core->value / mesh_ncol_;
    int src_core_col = src_core->value % mesh_ncol_;
    auto read_cores = Array<Integer>{src_core};
    Array<Integer> write_cores;
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
        write_cores.push_back(Integer(src_core_row * mesh_ncol_ + j));
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
        write_cores.push_back(Integer(i * mesh_ncol_ + src_core_col));
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

    for (const auto &op_node : collector.ops) {
      // Pre-assign a stable token id for each async site in this loop.
      // This lets the body rewriter attach the same token id every iteration,
      // enabling consistent loop-carried dependency reasoning.
      int token = GetNextTokenId();
      pre_assigned_tokens_[op_node] = token;

      // Keep a back-reference from token -> call for special handling after we
      // finish rewriting the loop (e.g. broadcast barrier initialization).
      const CallNode *call = op_node->value.as<CallNode>();
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

    for (const auto &p : collector.writes) {
      Array<ObjectRef> buffer_ref = {p.second->buffer, p.second->region};
      scope.writes.push_back(buffer_ref);
      scope.buffer_ref_to_token.Set(buffer_ref, pre_assigned_tokens_[p.first]);
    }
    for (const auto &p : collector.reads) {
      Array<ObjectRef> buffer_ref = {p.second->buffer, p.second->region};
      scope.reads.push_back(buffer_ref);
      scope.buffer_ref_to_token.Set(buffer_ref, pre_assigned_tokens_[p.first]);
    }

    // Push this loop scope so nested visitors can consult it when analyzing
    // read/write accesses inside the loop body.
    loop_scopes_.push_back(scope);

    Stmt loop_stmt = StmtMutator::VisitStmt_(op);

    scope = loop_scopes_.back();
    loop_scopes_.pop_back();
    for (const auto &op_node : collector.ops) {
      pre_assigned_tokens_.erase(op_node);
    }

    for (int token : scope.waited_tokens) {
      stmts.push_back(Evaluate(Call(DataType::Handle(), sync_null_token(),
                                    {IntImm(DataType::Int(32), token)})));
      if (scope.token_to_call.count(token)) {
        const CallNode *call = scope.token_to_call[token];
        if (call && call->op.same_as(broadcast_())) {
          int barrier_id = token_to_barrier_map[token];
          process_broadcast_barrier(call, token, barrier_id, stmts);
        }
      }
    }

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

        token_process_read_buffer(NormalizeToBufferRegion(call->args[0]), stmts,
                                  curr_token_id);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[1]), stmts,
                                  curr_token_id);
        token_process_read_buffer(NormalizeToBufferRegion(call->args[2]), stmts,
                                  curr_token_id, false);
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

    for (const auto &op_node : collector.ops) {
      int token = GetNextTokenId();
      pre_assigned_tokens_[op_node] = token;

      const CallNode *call = op_node->value.as<CallNode>();
      scope.token_to_call[token] = call;

      // check if it is a broadcast
      if (call && call->op.same_as(broadcast_())) {
        int barrier = GetNextBarrierId();
        token_to_barrier_map.Set(token, barrier);
        barrier_to_token_map.Set(barrier, token);
      }
    }

    for (const auto &p : collector.writes) {
      Array<ObjectRef> buffer_ref = {p.second->buffer, p.second->region};
      scope.writes.push_back(buffer_ref);
      scope.buffer_ref_to_token.Set(buffer_ref, pre_assigned_tokens_[p.first]);
    }

    for (const auto &p : collector.reads) {
      Array<ObjectRef> buffer_ref = {p.second->buffer, p.second->region};
      scope.reads.push_back(buffer_ref);
      scope.buffer_ref_to_token.Set(buffer_ref, pre_assigned_tokens_[p.first]);
    }

    loop_scopes_.push_back(scope);

    Stmt loop_stmt = StmtMutator::VisitStmt_(loop);

    scope = loop_scopes_.back();
    loop_scopes_.pop_back();
    for (const auto &op_node : collector.ops) {
      pre_assigned_tokens_.erase(op_node);
    }

    for (int token : scope.waited_tokens) {
      stmts.push_back(Evaluate(Call(DataType::Handle(), sync_null_token(),
                                    {IntImm(DataType::Int(32), token)})));
      if (scope.token_to_call.count(token)) {
        const CallNode *call = scope.token_to_call[token];
        if (call && call->op.same_as(broadcast_())) {
          int barrier_id = token_to_barrier_map[token];
          process_broadcast_barrier(call, token, barrier_id, stmts);
        }
      }
    }

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

// Rewriter to analyze and manage barrier synchronizations.
// Ensures that barriers initialized in branches are properly waited on,
// potentially hoisting waits or handling control flow implications.
class BarrierExtractRewriter : public StmtMutator {
public:
  BarrierExtractRewriter(Map<int, int> barrier_to_token_map)
      : barrier_to_token_map_(barrier_to_token_map) {
    barrier_init_map_ = Map<int, int>();
    barrier_init_ids_ = {};
    barrier_wait_ids_ = {};
  }

  std::vector<int> get_barrier_init_ids() const { return barrier_init_ids_; }
  std::vector<int> get_barrier_wait_ids() const { return barrier_wait_ids_; }

private:
  // Extracts barrier_init and barrier_arrive_and_wait calls, keeping track of
  // their presence within the current control flow block to inform dependency
  // analysis.
  Stmt VisitStmt_(const EvaluateNode *op) {
    const CallNode *call = op->value.as<CallNode>();
    if (call) {
      if (call->op.same_as(barrier_init())) {
        int barrier_id = call->args[0].as<IntImm>().value()->value;
        barrier_init_map_.Set(barrier_id, 1);
        barrier_init_ids_.push_back(barrier_id);
        return StmtMutator::VisitStmt_(op);
      } else if (call->op.same_as(barrier_arrive_and_wait())) {
        int barrier_id = call->args[0].as<IntImm>().value()->value;
        if (std::find(barrier_init_ids_.begin(), barrier_init_ids_.end(),
                      barrier_id) == barrier_init_ids_.end()) {
          if (std::find(barrier_wait_ids_.begin(), barrier_wait_ids_.end(),
                        barrier_id) == barrier_wait_ids_.end()) {
            // if the barrier wait does not have a corresponding barrier init in
            // current scope and  the barrier wait id is not in the
            // barrier_wait_ids_ list, we need to keep it and add its id to the
            // barrier_wait_ids_ list
            barrier_wait_ids_.push_back(barrier_id);
            return StmtMutator::VisitStmt_(op);
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  // Handles control flow branching for barrier synchronization.
  // Analyzes then/else cases separately to ensure any barriers initialized
  // inside the branches are appropriately waited on or hoisted to the parent
  // scope.
  Stmt VisitStmt_(const IfThenElseNode *op) {
    Array<Stmt> stmts;
    auto barrier_init_then_rewriter =
        BarrierExtractRewriter(barrier_to_token_map_);
    Stmt then_case = barrier_init_then_rewriter(op->then_case);

    auto then_barrier_init_ids =
        barrier_init_then_rewriter.get_barrier_init_ids();
    auto then_barrier_wait_ids =
        barrier_init_then_rewriter.get_barrier_wait_ids();

    for (int barrier_id : then_barrier_wait_ids) {
      if (std::find(barrier_init_ids_.begin(), barrier_init_ids_.end(),
                    barrier_id) == barrier_init_ids_.end()) {
        // If the barrier wait does not have a corresponding barrier init in the
        // current scope, we need to keep it and add its ID to the
        // barrier_wait_ids_ list for further processing.
        barrier_wait_ids_.push_back(barrier_id);
      } else {
        // if the barrier wait has a corresponding barrier init in current
        // scope, we need to wait on the token associated with the barrier init
        stmts.push_back(Evaluate(Call(
            DataType::Handle(), wait_token(),
            {IntImm(DataType::Int(32), barrier_to_token_map_[barrier_id])})));
        stmts.push_back(
            Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                          {IntImm(DataType::Int(32), barrier_id)})));
      }
    }
    for (int barrier_id : then_barrier_init_ids) {
      if (std::find(barrier_init_ids_.begin(), barrier_init_ids_.end(),
                    barrier_id) == barrier_init_ids_.end()) {
        barrier_init_ids_.push_back(barrier_id);
      }
    }

    Stmt else_case;
    if (op->else_case.defined()) {
      auto barrier_init_else_rewriter =
          BarrierExtractRewriter(barrier_to_token_map_);
      else_case = barrier_init_else_rewriter(op->else_case.value());
      auto else_barrier_init_ids =
          barrier_init_else_rewriter.get_barrier_init_ids();
      auto else_barrier_wait_ids =
          barrier_init_else_rewriter.get_barrier_wait_ids();
      for (int barrier_id : else_barrier_wait_ids) {
        if (std::find(barrier_init_ids_.begin(), barrier_init_ids_.end(),
                      barrier_id) == barrier_init_ids_.end()) {
          barrier_wait_ids_.push_back(barrier_id);
        } else {
          stmts.push_back(Evaluate(Call(
              DataType::Handle(), wait_token(),
              {IntImm(DataType::Int(32), barrier_to_token_map_[barrier_id])})));
          stmts.push_back(
              Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
                            {IntImm(DataType::Int(32), barrier_id)})));
        }
      }
      for (int barrier_id : else_barrier_init_ids) {
        if (std::find(barrier_init_ids_.begin(), barrier_init_ids_.end(),
                      barrier_id) == barrier_init_ids_.end()) {
          barrier_init_ids_.push_back(barrier_id);
        }
      }
    }

    stmts.push_back(IfThenElse(op->condition, then_case, else_case));
    return SeqStmt::Flatten(stmts);
  }

private:
  std::vector<int> barrier_init_ids_;
  std::vector<int> barrier_wait_ids_;

  Map<int, int> barrier_init_map_;

  Map<int, int> barrier_to_token_map_;
};

// Rewriter to inject final synchronization waits at the end of the device
// execution scope. This ensures all pending asynchronous operations are
// completed before the device kernel finishes.
class DeviceScopeWaitRewriter : public StmtMutator {
public:
  DeviceScopeWaitRewriter(Map<int, int> token_to_barrier_map)
      : token_to_barrier_map_(std::move(token_to_barrier_map)) {}

  Stmt VisitStmt_(const BlockNode *op) final {
    // If the block is the device root block (e.g., "tilelang_root" or "root" in
    // lowered IR without SplitHostDevice) we should apply the exit wait logic.
    // For typical tilelang lowering, "tilelang_root" represents the device
    // scope.
    if (op->name_hint == "tilelang_root" || op->name_hint == "root") {
      Stmt new_body = StmtMutator::VisitStmt(op->body);

      DeviceTokenCollector collector;
      collector(op->body);

      if (collector.tokens.empty()) {
        if (new_body.same_as(op->body)) {
          return ffi::GetRef<Stmt>(op);
        }
        auto n = CopyOnWrite(op);
        n->body = new_body;
        return Stmt(n);
      }

      Array<Stmt> stmts;
      if (const auto *seq = new_body.as<SeqStmtNode>()) {
        stmts = seq->seq;
      } else {
        stmts.push_back(new_body);
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

      auto n = CopyOnWrite(op);
      n->body = SeqStmt::Flatten(stmts);
      return Stmt(n);
    }
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
// determine if additional waits are required.
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

// Optimization pass to remove redundant synchronization calls.
// If a token or barrier has already been waited on in the current execution
// path, subsequent waits are unnecessary.
class EliminateRedundancyRewriter : public StmtMutator {
public:
  EliminateRedundancyRewriter(std::vector<int> parent_token_ids = {},
                              std::vector<int> parent_barrier_ids = {},
                              Map<int, int> barrier_to_token_map = {})
      : parent_token_ids_(std::move(parent_token_ids)),
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
  void PropagateResolvedStates(const Stmt &block) {
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
        get_all_token_ids(), get_all_barrier_ids(), barrier_to_token_map_);
    auto then_case = eliminate_sync_then_rewriter(op->then_case);

    Stmt else_case;
    if (op->else_case.defined()) {
      auto eliminate_sync_else_rewriter = EliminateRedundancyRewriter(
          get_all_token_ids(), get_all_barrier_ids(), barrier_to_token_map_);
      else_case = eliminate_sync_else_rewriter(op->else_case.value());
    }

    PropagateResolvedStates(ffi::GetRef<Stmt>(op));

    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const ForNode *op) {
    auto eliminate_sync_loop_rewriter = EliminateRedundancyRewriter(
        get_all_token_ids(), get_all_barrier_ids(), barrier_to_token_map_);
    auto body = eliminate_sync_loop_rewriter(op->body);

    PropagateResolvedStates(ffi::GetRef<Stmt>(op));

    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations);
  }

  Stmt VisitStmt_(const WhileNode *op) {
    auto eliminate_sync_loop_rewriter = EliminateRedundancyRewriter(
        get_all_token_ids(), get_all_barrier_ids(), barrier_to_token_map_);
    auto body = eliminate_sync_loop_rewriter(op->body);

    PropagateResolvedStates(ffi::GetRef<Stmt>(op));

    return While(op->condition, body);
  }

private:
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
        InjectSyncRewriter(f->buffer_map, mesh_nrow, mesh_ncol);
    f.CopyOnWrite()->body = inject_sync_rewriter(f->body);

    auto barrier_extract_rewriter =
        BarrierExtractRewriter(inject_sync_rewriter.get_barrier_to_token_map());
    f.CopyOnWrite()->body = barrier_extract_rewriter(f->body);

    auto device_scope_wait_rewriter = DeviceScopeWaitRewriter(
        inject_sync_rewriter.get_token_to_barrier_map());
    f.CopyOnWrite()->body = device_scope_wait_rewriter(f->body);

    auto eliminate_redundancy_rewriter = EliminateRedundancyRewriter(
        std::vector<int>({}), std::vector<int>({}),
        inject_sync_rewriter.get_barrier_to_token_map());
    f.CopyOnWrite()->body = eliminate_redundancy_rewriter(f->body);

    return f;
  }
};

// TVM transform pass entry point.
// Applies the SunmmioSyncRewriter to inject required synchronization
// primitives.
tvm::transform::Pass InjectSunmmioSync() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
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
