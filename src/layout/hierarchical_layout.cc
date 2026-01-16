/*!
 * \file layout/hierarchical_layout.cc
 * \brief Define Hierarchical Layout and related functions
 *
 */

#ifndef TVM_TL_LAYOUT_HIERARCHICAL_H_
#define TVM_TL_LAYOUT_HIERARCHICAL_H_

#include "layout.h"
#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

// Helper to decompose a logical index into hierarchical indices (as PrimExpr)
Array<PrimExpr> decompose_index_expr(PrimExpr logical_idx,
                                     const Array<Integer> &factors_arr) {
  Array<PrimExpr> h_indices;
  PrimExpr rem = logical_idx;
  for (int i = factors_arr.size() - 1; i >= 0; --i) {
    PrimExpr f = factors_arr[i];
    h_indices.push_back(FloorMod(rem, f));
    rem = FloorDiv(rem, f);
  }
  // The results are collected in reverse order, so reverse them to match Python
  Array<PrimExpr> result_h_indices;
  for (int i = h_indices.size() - 1; i >= 0; --i) {
    result_h_indices.push_back(h_indices[i]);
  }
  return result_h_indices;
}

Layout makeHierarchicalLayout(Array<Integer> hdims_arr,
                              Array<Integer> hstrides_arr,
                              Array<Array<Integer>> groups_arr,
                              Array<Integer> logical_shape_arr) {
  ICHECK_EQ(hdims_arr.size(), hstrides_arr.size())
      << "hdims and hstrides must have the same length";
  ICHECK_EQ(groups_arr.size(), logical_shape_arr.size())
      << "Number of groups must match logical shape dimensions";

  Array<IterVar> input_vars;
  for (size_t i = 0; i < logical_shape_arr.size(); ++i) {
    input_vars.push_back(
        make_itervar(std::string{'_', char('i' + i)}, logical_shape_arr[i]));
  }

  // The final physical offset expression
  PrimExpr total_offset =
      make_zero(DataType::Int(32)); // Assuming int32 for offset

  int h_idx_offset_global =
      0; // Tracks position in the flat hdims/hstrides arrays

  for (size_t logical_dim_idx = 0; logical_dim_idx < groups_arr.size();
       ++logical_dim_idx) {
    Array<Integer> group = groups_arr[logical_dim_idx];
    ICHECK_EQ(group.size(), 2) << "Group must contain 2 elements (start, end)";
    int group_start = group[0].IntValue();
    int group_end = group[1].IntValue();
    int group_len = group_end - group_start;

    // Extract factors (hdims) and strides (hstrides) for this logical dimension
    Array<Integer> factors_for_dim;
    Array<Integer> strides_for_dim;
    for (int i = 0; i < group_len; ++i) {
      factors_for_dim.push_back(hdims_arr[group_start + i]);
      strides_for_dim.push_back(hstrides_arr[group_start + i]);
    }

    // Get the logical input variable for this dimension
    PrimExpr logical_input_var = input_vars[logical_dim_idx]->var;

    // Decompose the logical input variable into hierarchical indices for this
    // dim
    Array<PrimExpr> h_indices_for_dim =
        decompose_index_expr(logical_input_var, factors_for_dim);

    // Calculate the offset contribution from this logical dimension
    PrimExpr dim_offset_contribution = make_zero(DataType::Int(32));
    ICHECK_EQ(h_indices_for_dim.size(), strides_for_dim.size())
        << "Hierarchical indices and strides size mismatch for logical "
           "dimension "
        << logical_dim_idx;

    for (size_t i = 0; i < h_indices_for_dim.size(); ++i) {
      dim_offset_contribution =
          dim_offset_contribution + h_indices_for_dim[i] * strides_for_dim[i];
    }
    total_offset = total_offset + dim_offset_contribution;
  }

  return Layout(input_vars, {total_offset});
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_HIERARCHICAL_H_
