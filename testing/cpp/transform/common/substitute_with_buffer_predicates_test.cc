#include <gtest/gtest.h>

#include <tvm/node/structural_equal.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include "transform/common/substitute_with_buffer_predicates.h"

namespace tvm {
namespace tl {
namespace {

using namespace tir;

PrimExpr I(int value) { return IntImm(DataType::Int(32), value); }

TEST(SubstituteWithBufferPredicatesTest, RewritesLoadAndStorePredicates) {
  Var old_j("old_j", DataType::Int(32));
  Var new_j("new_j", DataType::Int(32));
  Var kj("kj", DataType::Int(32));
  Buffer src = decl_buffer({I(4), I(32)}, DataType::Float(32), "src");
  Buffer dst = decl_buffer({I(4), I(32)}, DataType::Float(32), "dst");

  PrimExpr row = I(0);
  PrimExpr col = old_j * I(32) + kj;
  PrimExpr predicate = col < I(4);
  BufferLoad load(src, {row, col}, predicate);
  Stmt store = BufferStore(dst, load, {row, col}, predicate);

  Stmt rewritten = SubstituteWithBufferPredicates(store, {{old_j, new_j}});
  const auto *store_node = rewritten.as<BufferStoreNode>();
  ASSERT_NE(store_node, nullptr);
  const auto *load_node = store_node->value.as<BufferLoadNode>();
  ASSERT_NE(load_node, nullptr);
  ASSERT_TRUE(store_node->predicate.defined());
  ASSERT_TRUE(load_node->predicate.defined());

  PrimExpr expected_col = new_j * I(32) + kj;
  PrimExpr expected_predicate = expected_col < I(4);
  StructuralEqual equal;
  EXPECT_TRUE(equal(store_node->indices[1], expected_col));
  EXPECT_TRUE(equal(load_node->indices[1], expected_col));
  EXPECT_TRUE(equal(store_node->predicate.value(), expected_predicate));
  EXPECT_TRUE(equal(load_node->predicate.value(), expected_predicate));
}

} // namespace
} // namespace tl
} // namespace tvm
