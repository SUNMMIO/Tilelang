#ifndef AST_TRAVERSER_H
#define AST_TRAVERSER_H

#include "../../op/builtin.h"
#include "../../op/utils.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/stmt.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

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

class ASTTraverser : public StmtVisitor {
public:
  ASTTraverser(const PrimFunc &f) {
    for (const auto &[_, buffer] : f->buffer_map) {
      this->buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
  }

  std::pair<Array<BufferRegion>, Array<BufferRegion>>
  buffer_region_collector(const PrimExpr &expr) {
    auto buf_load_collector = BufferAccessCollector(buffer_data_to_buffer_);
    buf_load_collector(expr);
    Array<BufferRegion> read_regions = buf_load_collector.GetReads();
    Array<BufferRegion> write_regions = buf_load_collector.GetWrites();
    return {read_regions, write_regions};
  }

  void VisitStmt_(const AttrStmtNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->value);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->value);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) {
    auto [min_read_regions, min_write_regions] =
        buffer_region_collector(op->min);
    for (auto it : min_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : min_write_regions) {
      write_buffer_regions_.insert(it);
    }

    auto [extent_read_regions, extent_write_regions] =
        buffer_region_collector(op->extent);
    for (auto it : extent_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : extent_write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->condition);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->condition);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->condition);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AssertStmtNode *op) {
    auto [condition_read_regions, condition_write_regions] =
        buffer_region_collector(op->condition);
    for (auto it : condition_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : condition_write_regions) {
      write_buffer_regions_.insert(it);
    }

    auto [message_read_regions, message_write_regions] =
        buffer_region_collector(op->message);
    for (auto it : message_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : message_write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockRealizeNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->predicate);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) {
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
    write_buffer_regions_.insert(store_region);

    // For a store statement, we also need to check the read dependencies for
    // the value to be stored. For example, in the statement A[i] = B[j] + C[k],
    // we need to check the read dependencies for the buffers B and C.
    auto [read_regions, write_regions] = buffer_region_collector(op->value);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) {
    const CallNode *call = op->value.as<CallNode>();
    if (call->op.same_as(dma_copy())) {
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[0]));
      write_buffer_regions_.insert(NormalizeToBufferRegion(call->args[1]));
    } else if (call->op.same_as(mma_sunmmio())) {
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[0]));
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[1]));
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[2]));

      write_buffer_regions_.insert(NormalizeToBufferRegion(call->args[2]));
      // } else if (call->op.same_as(broadcast_())) {
      //   read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[0]));
      //   write_buffer_regions_.insert(NormalizeToBufferRegion(call->args[1]));
    } else {
      auto [read_regions, write_regions] = buffer_region_collector(op->value);
      for (auto it : read_regions) {
        read_buffer_regions_.insert(it);
      }
      for (auto it : write_regions) {
        write_buffer_regions_.insert(it);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    StmtVisitor::VisitStmt_(op);
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
  }

  void clear() {
    read_buffer_regions_.clear();
    write_buffer_regions_.clear();
  }

  void traverse_stmt(Stmt stmt) {
    clear();
    VisitStmt(stmt);
  }

  void traverse_expr(PrimExpr expr) {
    clear();
    buffer_region_collector(expr);
  }

public:
  Map<Var, Buffer> buffer_data_to_buffer_;

  std::set<BufferRegion> read_buffer_regions_;
  std::set<BufferRegion> write_buffer_regions_;
};

} // namespace tl
} // namespace tvm

#endif
