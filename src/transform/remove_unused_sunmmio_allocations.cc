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
 * \file remove_unused_sunmmio_allocations.cc
 * \brief Remove SunMMIO allocations that survive RemoveNoOp only because of
 *        metadata annotations.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>
#include <utility>

#include "../layout/layout.h"
#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

class ExecutableBufferUseCollector : public StmtExprVisitor {
public:
  static std::unordered_set<const VarNode *> Collect(const Stmt &body) {
    ExecutableBufferUseCollector collector;
    collector(body);
    return std::move(collector.used_vars_);
  }

private:
  void VisitStmt_(const AllocateNode *op) final {
    for (const PrimExpr &extent : op->extents) {
      VisitExpr(extent);
    }
    VisitExpr(op->condition);
    VisitStmt(op->body);
  }

  void VisitStmt_(const DeclBufferNode *op) final { VisitStmt(op->body); }

  void VisitExpr_(const VarNode *op) final { used_vars_.insert(op); }

  void VisitExpr_(const BufferLoadNode *op) final {
    used_vars_.insert(op->buffer->data.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    used_vars_.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_set<const VarNode *> used_vars_;
};

class UnusedSunmmioAllocationRemover : public StmtExprMutator {
public:
  explicit UnusedSunmmioAllocationRemover(
      const std::unordered_set<const VarNode *> &used_vars)
      : used_vars_(used_vars) {}

private:
  bool IsUsed(const Var &var) const { return used_vars_.count(var.get()) != 0; }

  Stmt VisitStmt_(const AllocateNode *op) final {
    Stmt body = VisitStmt(op->body);
    if (!IsUsed(op->buffer_var)) {
      return body;
    }
    Array<PrimExpr> extents;
    for (const PrimExpr &extent : op->extents) {
      extents.push_back(VisitExpr(extent));
    }
    PrimExpr condition = VisitExpr(op->condition);
    return Allocate(op->buffer_var, op->dtype, std::move(extents),
                    std::move(condition), std::move(body), op->annotations,
                    op->span);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    Stmt body = VisitStmt(op->body);
    if (!IsUsed(op->buffer->data)) {
      return body;
    }
    return DeclBuffer(op->buffer, std::move(body), op->span);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto layout_it = block->annotations.find(attr::kLayoutMap);
    if (layout_it == block->annotations.end()) {
      return block;
    }

    Map<String, ffi::Any> annotations = block->annotations;
    if (auto layout_map = (*layout_it).second.as<Map<Buffer, Layout>>()) {
      Map<Buffer, Layout> filtered;
      for (const auto &[buffer, layout] : layout_map.value()) {
        if (IsUsed(buffer->data)) {
          filtered.Set(buffer, layout);
        }
      }
      if (filtered.empty()) {
        annotations.erase(attr::kLayoutMap);
      } else {
        annotations.Set(attr::kLayoutMap, filtered);
      }
    } else if (auto layout_map = (*layout_it).second.as<Map<Var, Layout>>()) {
      Map<Var, Layout> filtered;
      for (const auto &[var, layout] : layout_map.value()) {
        if (IsUsed(var)) {
          filtered.Set(var, layout);
        }
      }
      if (filtered.empty()) {
        annotations.erase(attr::kLayoutMap);
      } else {
        annotations.Set(attr::kLayoutMap, filtered);
      }
    }
    block.CopyOnWrite()->annotations = std::move(annotations);
    return block;
  }

  const std::unordered_set<const VarNode *> &used_vars_;
};

PrimFunc RemoveUnusedSunmmioAllocationsFromFunc(PrimFunc func) {
  std::unordered_set<const VarNode *> used_vars =
      ExecutableBufferUseCollector::Collect(func->body);
  UnusedSunmmioAllocationRemover remover(used_vars);
  func.CopyOnWrite()->body = remover(func->body);

  if (auto layout_map = func->GetAttr<Map<Buffer, Layout>>(attr::kLayoutMap)) {
    Map<Buffer, Layout> filtered;
    for (const auto &[buffer, layout] : layout_map.value()) {
      if (used_vars.count(buffer->data.get())) {
        filtered.Set(buffer, layout);
      }
    }
    if (filtered.empty()) {
      func = WithoutAttr(std::move(func), ffi::String(attr::kLayoutMap));
    } else {
      func = WithAttr(std::move(func), attr::kLayoutMap, filtered);
    }
  }

  if (auto ping_pong =
          func->GetAttr<Map<Var, String>>(tl::attr::kSunmmioAllocPingPong)) {
    Map<Var, String> filtered;
    for (const auto &[var, bank] : ping_pong.value()) {
      if (used_vars.count(var.get())) {
        filtered.Set(var, bank);
      }
    }
    if (filtered.empty()) {
      func = WithoutAttr(std::move(func),
                         ffi::String(tl::attr::kSunmmioAllocPingPong));
    } else {
      func =
          WithAttr(std::move(func), tl::attr::kSunmmioAllocPingPong, filtered);
    }
  }
  return func;
}

tvm::transform::Pass RemoveUnusedSunmmioAllocations() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    return RemoveUnusedSunmmioAllocationsFromFunc(std::move(func));
  };
  return tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.RemoveUnusedSunmmioAllocations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RemoveUnusedSunmmioAllocations",
                        RemoveUnusedSunmmioAllocations);
}

} // namespace tl
} // namespace tvm
