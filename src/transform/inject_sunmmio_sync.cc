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
 * \file tma_barrier_rewriter.cc
 * \brief Rewrite TMA barriers for cuda GPU (sm90+)
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

#include <utility>

#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/utils.h"
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
  Array<BufferRegion> GetWrites() const { return writes_; }

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
    }
    // else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    //   ICHECK(args[0].as<BufferLoadNode>());
    //   Buffer mbar_buf = args[0].as<BufferLoadNode>()->buffer;
    //   auto buffer_reads =
    //       chain_builder_.mbar_to_buffer_reads_.find(mbar_buf.get());
    //   auto buffer_writes =
    //       chain_builder_.mbar_to_buffer_writes_.find(mbar_buf.get());
    //   if (buffer_reads != chain_builder_.mbar_to_buffer_reads_.end()) {
    //     reads_.insert(reads_.end(), buffer_reads->second.begin(),
    //                   buffer_reads->second.end());
    //   }
    //   if (buffer_writes != chain_builder_.mbar_to_buffer_writes_.end()) {
    //     writes_.insert(
    //         writes_.end(),
    //         chain_builder_.mbar_to_buffer_writes_.at(mbar_buf.get()).begin(),
    //         chain_builder_.mbar_to_buffer_writes_.at(mbar_buf.get()).end());
    //   }
    // }

    else {
      ExprVisitor::VisitExpr_(op);
    }
  }

private:
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  Map<Var, Buffer> buffer_data_to_buffer_;
};

// Collector for asynchronous write operations within a loop body.
// Identifies DMA copies, MMA operations, and Broadcasts that happen
// asynchronously.
class LoopAsyncWriteCollector : public StmtVisitor {
public:
  void VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call = op->value.as<CallNode>();
    if (call) {
      if (call->op.same_as(dma_copy())) {
        writes.push_back({op, NormalizeToBufferRegion(call->args[1])});
      } else if (call->op.same_as(mma_sunmmio())) {
        writes.push_back({op, NormalizeToBufferRegion(call->args[2])});
      } else if (call->op.same_as(broadcast_())) {
        writes.push_back({op, NormalizeToBufferRegion(call->args[1])});
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  std::vector<std::pair<const EvaluateNode *, BufferRegion>> writes;
};

// Represents the scope of a loop for dependency tracking.
// Stores writes that happen within the loop to check for loop-carried
// dependencies.
struct LoopScope {
  Array<Array<ObjectRef>> writes;
  Map<Array<ObjectRef>, int> token_map;
  Var loop_var;
  PrimExpr min;
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
        Buffer buf_buffer = Downcast<Buffer>(buf[0]);
        Region buf_region = Downcast<Region>(buf[1]);
        if (src_buffer.same_as(buf_buffer) &&
            RegionIntersect(src_region, buf_region)) {
          int token = scope.token_map[buf];
          if (waited_tokens.count(token) == 0) {
            process_wait_token_and_barrier_wait(stmts, token);
            waited_tokens.insert(token);
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
        process_wait_token_and_barrier_wait(stmts, read_buffer_token_map[buf]);
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
        process_wait_token_and_barrier_wait(stmts, write_buffer_token_map[buf]);
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

  void token_process_prim_expr(const PrimExpr &expr, Array<Stmt> &stmts) {
    auto buf_load_collector = BufferAccessCollector(buffer_data_to_buffer_);
    buf_load_collector(expr);
    Array<BufferRegion> read_regions = buf_load_collector.GetReads();
    for (const auto &read_region : read_regions) {
      token_process_read_buffer(read_region, stmts, -1, false);
    }
    Array<BufferRegion> write_regions = buf_load_collector.GetWrites();
    for (const auto &write_region : write_regions) {
      token_process_write_buffer(write_region, stmts, -1, false);
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

    LoopAsyncWriteCollector collector;
    collector(op->body);

    LoopScope scope;

    for (const auto &p : collector.writes) {
      int token = GetNextTokenId();
      pre_assigned_tokens_[p.first] = token;

      // check if it is a broadcast
      const CallNode *call = p.first->value.as<CallNode>();
      if (call && call->op.same_as(broadcast_())) {
        int barrier = GetNextBarrierId();
        token_to_barrier_map.Set(token, barrier);
        barrier_to_token_map.Set(barrier, token);
      }

      Array<ObjectRef> buffer_ref = {p.second->buffer, p.second->region};
      scope.writes.push_back(buffer_ref);
      scope.token_map.Set(buffer_ref, token);
    }

    loop_scopes_.push_back(scope);

    Stmt loop_stmt = StmtMutator::VisitStmt_(op);

    loop_scopes_.pop_back();
    for (const auto &p : collector.writes) {
      pre_assigned_tokens_.erase(p.first);
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

        token_process_read_buffer(NormalizeToBufferRegion(call->args[0]), stmts,
                                  curr_token_id);
        token_process_write_buffer(NormalizeToBufferRegion(call->args[1]),
                                   stmts, curr_token_id);

        curr_stmt_with_token_id(call, stmts, curr_token_id);

        init_barrier_(stmts, curr_barrier_id, curr_token_id, src_core,
                      write_cores);

        // stmts.push_back(
        //     Evaluate(Call(DataType::Handle(), wait_token(),
        //                   {IntImm(DataType::Int(32), curr_token_id)})));

        // stmts.push_back(
        //     Evaluate(Call(DataType::Handle(), barrier_arrive_and_wait(),
        //                   {IntImm(DataType::Int(32), curr_barrier_id)})));

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

    LoopAsyncWriteCollector collector;
    collector(loop->body);

    LoopScope scope;
    scope.loop_var = loop->loop_var;
    scope.min = loop->min;

    for (const auto &p : collector.writes) {
      int token = GetNextTokenId();
      pre_assigned_tokens_[p.first] = token;

      // check if it is a broadcast
      const CallNode *call = p.first->value.as<CallNode>();
      if (call && call->op.same_as(broadcast_())) {
        int barrier = GetNextBarrierId();
        token_to_barrier_map.Set(token, barrier);
        barrier_to_token_map.Set(barrier, token);
      }

      Array<ObjectRef> buffer_ref = {p.second->buffer, p.second->region};
      scope.writes.push_back(buffer_ref);
      scope.token_map.Set(buffer_ref, token);
    }

    loop_scopes_.push_back(scope);

    Stmt loop_stmt = StmtMutator::VisitStmt_(loop);

    loop_scopes_.pop_back();
    for (const auto &p : collector.writes) {
      pre_assigned_tokens_.erase(p.first);
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
            // current scope, we need to keep it and add its id to the
            // barrier_wait_ids_ list
            barrier_wait_ids_.push_back(barrier_id);
            return StmtMutator::VisitStmt_(op);
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

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
      LOG(INFO) << "barrier wait id in then case: " << barrier_id;
      for (int init_id : barrier_init_ids_) {
        LOG(INFO) << "barrier init id in current scope: " << init_id;
      }
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
    auto then_barrier_ids =
        eliminate_sync_then_rewriter.get_current_barrier_ids();
    for (int barrier_id : then_barrier_ids) {
      if (std::find(current_barrier_ids_.begin(), current_barrier_ids_.end(),
                    barrier_id) == current_barrier_ids_.end()) {
        current_barrier_ids_.push_back(barrier_id);
        current_token_ids_.push_back(barrier_to_token_map_[barrier_id]);
      }
    }

    Stmt else_case;
    if (op->else_case.defined()) {
      auto eliminate_sync_else_rewriter = EliminateRedundancyRewriter(
          get_all_token_ids(), get_all_barrier_ids(), barrier_to_token_map_);
      else_case = eliminate_sync_else_rewriter(op->else_case.value());
      auto else_barrier_ids =
          eliminate_sync_else_rewriter.get_current_barrier_ids();
      for (int barrier_id : else_barrier_ids) {
        if (std::find(current_barrier_ids_.begin(), current_barrier_ids_.end(),
                      barrier_id) == current_barrier_ids_.end()) {
          current_barrier_ids_.push_back(barrier_id);
          current_token_ids_.push_back(barrier_to_token_map_[barrier_id]);
        }
      }
    }
    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const ForNode *op) {
    auto eliminate_sync_then_rewriter = EliminateRedundancyRewriter(
        get_all_token_ids(), get_all_barrier_ids(), barrier_to_token_map_);
    auto body = eliminate_sync_then_rewriter(op->body);
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations);
  }

  Stmt VisitStmt_(const WhileNode *op) {
    auto eliminate_sync_then_rewriter = EliminateRedundancyRewriter(
        get_all_token_ids(), get_all_barrier_ids(), barrier_to_token_map_);
    auto body = eliminate_sync_then_rewriter(op->body);
    return While(op->condition, body);
  }

private:
  std::vector<int> parent_token_ids_;
  std::vector<int> current_token_ids_;
  std::vector<int> parent_barrier_ids_;
  std::vector<int> current_barrier_ids_;
  Map<int, int> barrier_to_token_map_;
};

class SunmmioSyncRewriter : public IRMutatorWithAnalyzer {
public:
  SunmmioSyncRewriter(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  static PrimFunc Rewrite(PrimFunc f, arith::Analyzer *analyzer) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget).value();
    int mesh_nrow = get_target_mesh(target, 0);
    int mesh_ncol = get_target_mesh(target, 1);

    auto inject_sync_rewriter =
        InjectSyncRewriter(f->buffer_map, mesh_nrow, mesh_ncol);
    f.CopyOnWrite()->body = inject_sync_rewriter(f->body);

    auto barrier_extract_rewriter =
        BarrierExtractRewriter(inject_sync_rewriter.get_barrier_to_token_map());
    f.CopyOnWrite()->body = barrier_extract_rewriter(f->body);

    auto eliminate_redundancy_rewriter = EliminateRedundancyRewriter(
        std::vector<int>({}), std::vector<int>({}),
        inject_sync_rewriter.get_barrier_to_token_map());
    f.CopyOnWrite()->body = eliminate_redundancy_rewriter(f->body);

    return f;
  }
};

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
