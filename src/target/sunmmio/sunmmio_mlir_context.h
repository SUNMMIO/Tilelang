#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_

#include "sunmmio_mlir_type.h"

#include "../../layout/layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

struct SunmmioMlirContext {
  SunmmioMlirContext();

  using TirLayoutMap = ffi::Map<tir::Buffer, tl::Layout>;

  mlir::MLIRContext mlir_ctx;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  using MLIRValueTable = std::unordered_map<std::string, mlir::Value>;
  std::vector<MLIRValueTable> mlir_value_table_stack;

  std::unordered_map<int64_t, mlir::Value> token_by_id;
  std::unordered_map<int64_t, mlir::Value> barrier_by_mask;

  struct SavedToken {
    bool existed{false};
    mlir::Value value;
  };

  struct ForFrame {
    mlir::scf::ForOp op;
    ffi::Map<ffi::String, ffi::Any> annotations;
    std::vector<std::string> live_out_value_names;
    // Token ids that must be carried by scf.for and materialized after the
    // loop.
    std::vector<int64_t> live_out_token_ids;
    // token_id -> index into iter_tokens / produced_tokens.
    std::unordered_map<int64_t, int> token_id_to_index;
    // Region iter_args of scf.for (initially seeded by init_args).
    std::vector<mlir::Value> iter_tokens;
    // Tokens produced in the loop body; falls back to iter_tokens when empty.
    std::vector<mlir::Value> produced_tokens;
    // Snapshots of ctx.token_by_id to restore when closing the loop.
    std::unordered_map<int64_t, SavedToken> saved_token_by_id;
  };
  std::vector<ForFrame> for_stack;
  std::vector<TirLayoutMap> layout_map_stack;
  std::vector<TirLayoutMap> global_layout_map_stack;

  struct WhileFrame {
    mlir::scf::WhileOp op;
    bool in_body{false};
    // Token ids that must be carried by scf.while and materialized after the
    // loop.
    std::vector<int64_t> live_out_token_ids;
    // token_id -> index into iter_tokens / produced_tokens.
    std::unordered_map<int64_t, int> token_id_to_index;
    // Region arguments in scf.while before/after regions.
    std::vector<mlir::Value> before_tokens;
    std::vector<mlir::Value> iter_tokens;
    // Tokens produced in the loop body; falls back to iter_tokens when empty.
    std::vector<mlir::Value> produced_tokens;
    // Snapshots of ctx.token_by_id to restore when closing the loop.
    std::unordered_map<int64_t, SavedToken> saved_token_by_id;
  };
  std::vector<WhileFrame> while_stack;

  struct IfFrame {
    mlir::scf::IfOp op;
    bool in_else{false};
    std::vector<std::string> live_out_value_names;
    // Token ids that must be carried by scf.if and materialized after the if.
    std::vector<int64_t> live_out_token_ids;
    // token_id -> index into base_tokens / produced_tokens / then_yield_tokens.
    std::unordered_map<int64_t, int> token_id_to_index;
    // The token values seen before entering the if (used as defaults for both
    // branches).
    std::vector<mlir::Value> base_tokens;
    // Tokens produced by the active branch; merged into the surrounding scope
    // after EndIf.
    std::vector<mlir::Value> produced_tokens;
    // Tokens yielded from the then branch (cached to build consistent yields
    // across branches).
    std::vector<mlir::Value> then_yield_tokens;
    // Snapshots of ctx.token_by_id to restore when closing the if.
    std::unordered_map<int64_t, SavedToken> saved_token_by_id;
  };
  std::vector<IfFrame> if_stack;

  enum class ControlKind { kFor, kIf, kWhile };

  struct ControlNode {
    ControlKind kind;
    int index{0};
  };
  std::vector<ControlNode> control_flow_stack;

  const ffi::Map<ffi::String, ffi::Any> *CurrentForAnnotations() const {
    if (for_stack.empty()) {
      return nullptr;
    }
    return &for_stack.back().annotations;
  }

  void ClearMLIRValueScopes() { mlir_value_table_stack.clear(); }

  void PushMLIRValueScope() { mlir_value_table_stack.emplace_back(); }

  void PopMLIRValueScope() {
    if (!mlir_value_table_stack.empty()) {
      mlir_value_table_stack.pop_back();
    }
  }

  mlir::Value LookupMLIRValue(const std::string &name) const {
    for (auto it = mlir_value_table_stack.rbegin();
         it != mlir_value_table_stack.rend(); ++it) {
      auto vit = it->find(name);
      if (vit != it->end()) {
        return vit->second;
      }
    }
    return mlir::Value();
  }

  mlir::Value LookupOrCreateFakeValue(const SunMMIOValue &value,
                                      const std::string &debug_tag) {
    mlir::Value existing = LookupMLIRValue(value.value);
    if (existing) {
      return existing;
    }

    auto fake_op = mlir::arith::ConstantIntOp::create(
        builder, SunmmioMlirType(*this).MakeDebugLoc(debug_tag), 0, 32);
    std::string attr_str =
        debug_tag + (value.value.empty() ? "" : ":" + value.value);
    fake_op->setAttr("sunmmio.fake", builder.getStringAttr(attr_str));
    mlir::Value fake_value = fake_op.getResult();
    if (!value.value.empty()) {
      BindMLIRValue(value.value, fake_value);
    }
    return fake_value;
  }

  void BindMLIRValue(const std::string &name, mlir::Value v) {
    if (mlir_value_table_stack.empty()) {
      mlir_value_table_stack.emplace_back();
    }
    mlir_value_table_stack.back()[name] = v;
    for (auto it = control_flow_stack.rbegin(); it != control_flow_stack.rend();
         ++it) {
      if (it->kind == ControlKind::kFor) {
        ForFrame &frame = for_stack[it->index];
        auto vit = std::find(frame.live_out_value_names.begin(),
                             frame.live_out_value_names.end(), name);
        if (vit == frame.live_out_value_names.end()) {
          continue;
        }
        int idx = static_cast<int>(
            std::distance(frame.live_out_value_names.begin(), vit));
        if (idx >= 0 && idx < static_cast<int>(frame.produced_tokens.size())) {
          frame.produced_tokens[idx] = v;
        }
        break;
      }
      if (it->kind == ControlKind::kWhile) {
        continue;
      }
      IfFrame &frame = if_stack[it->index];
      auto vit = std::find(frame.live_out_value_names.begin(),
                           frame.live_out_value_names.end(), name);
      if (vit == frame.live_out_value_names.end()) {
        continue;
      }
      int idx = static_cast<int>(
          std::distance(frame.live_out_value_names.begin(), vit));
      if (idx >= 0 && idx < static_cast<int>(frame.produced_tokens.size())) {
        frame.produced_tokens[idx] = v;
      }
      break;
    }
  }

  void ClearLayoutScopes();
  void PushLayoutScope(const TirLayoutMap &layout_map,
                       const TirLayoutMap &global_layout_map);
  void PopLayoutScope();
  ffi::Optional<tl::Layout> LookupLayout(const tir::Buffer &buffer) const;
  void ApplyLayoutToType(const tir::Buffer &buffer, SunMMIOType *type) const;

  void Clear();
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
