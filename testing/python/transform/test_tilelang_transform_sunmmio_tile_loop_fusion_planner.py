import pytest

from testing.python.transform._sunmmio_tile_loop_fusion_test_utils import *


def test_sunmmio_tile_loop_fusion_phase1_guardrail_helpers():
    summary = get_phase1_guardrail_summary()

    limit = summary["planner_cost_limit"]
    assert summary["saturating_add_overflow"] == limit
    assert summary["saturating_mul_overflow"] == limit
    assert summary["resident_dedupe_identical_count"] == 1
    assert summary["resident_dedupe_payload_distinct_count"] == 2
    assert summary["resident_dedupe_instance_distinct_count"] == 2


def test_sunmmio_tile_loop_fusion_phase1_bitset_rejects_padding_bits():
    assert check_phase1_bitset_bounds(19, 18) is True
    with pytest.raises(tvm.error.TVMError):
        check_phase1_bitset_bounds(19, 19)


def test_sunmmio_tile_loop_fusion_raw_coverage_only_suppresses_exact_covered_uses():
    summary = get_raw_coverage_accounting_summary()

    assert summary == {
        "first_write_cut_cost": 64,
        "first_shared_read_cost": 80,
        "second_write_cut_cost": 0,
        "second_shared_read_cost": 0,
    }


def test_sunmmio_tile_loop_fusion_planner_large_window_falls_back_to_source_order():
    mod = IRModule.from_expr(long_row_chain_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tiles_lowering(mod)

    discovery = get_discovery_summary(mod)
    plan_summary = get_plan_summary(mod)
    plan = plan_summary["plans"][0]

    assert discovery["region_run_count"] == 1
    assert discovery["region_run_lengths"] == [19]
    assert plan_summary["plan_count"] == 1
    assert plan["order"] == list(range(19))
    assert all(action["close_to_depth"] == 0 for action in plan["actions"])
    assert all(action["open_to_depth"] == 0 for action in plan["actions"])
    assert not any(node["is_scope"] for node in walk_plan_tree(plan["tree"]))
    assert plan["score"]["write_cut_cost"] > 0


def test_sunmmio_tile_loop_fusion_planner_dense_flash_attention_online_softmax_chain():
    mod = IRModule.from_expr(flash_attention_online_softmax_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tiles_lowering(mod)

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    plan_summary = get_plan_summary(mod)
    (graph,) = dependence["graphs"]
    (plan,) = plan_summary["plans"]
    reduce_max_region = discovery["regions"][2]
    reduce_sum_region = discovery["regions"][6]

    assert discovery["region_count"] == 10
    assert discovery["region_run_count"] == 1
    assert discovery["region_run_lengths"] == [10]
    assert plan_summary["plan_count"] == 1
    assert {
        (0, 1, "WAR", "scores_max", 1, 0),
        (1, 2, "RAW", "scores_max", 1, 32),
        (2, 3, "RAW", "scores_max", 1, 32),
        (0, 3, "RAW", "scores_max_prev", 1, 256),
        (3, 4, "RAW", "scores_max", 1, 256),
        (3, 5, "RAW", "scores_max", 1, 32),
        (5, 6, "RAW", "acc_s", 2, 1024),
        (4, 7, "RAW", "scores_scale", 1, 256),
        (6, 7, "RAW", "scores_sum", 1, 32),
        (5, 8, "RAW", "acc_s", 2, 1024),
        (4, 9, "RAW", "scores_scale", 1, 32),
    } <= graph_edge_keys(graph)
    assert plan["region_indices"] == list(range(10))
    assert plan["order"] == [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]
    assert plan["score"] == {
        "write_cut_cost": 1792,
        "shared_read_cost": 12576,
        "live_range_penalty": 26624,
        "reorder_penalty": 2,
    }
    assert [region["execution_loop_extents"] for region in discovery["regions"]] == [
        ["1"],
        ["1"],
        ["8", "1"],
        ["1"],
        ["1"],
        ["8", "1"],
        ["8", "1"],
        ["1"],
        ["8", "1"],
        ["8", "1"],
    ]

    assert set(buffer_regions_by_name(reduce_max_region["use_in"])) == {"acc_s", "scores_max"}
    assert set(buffer_regions_by_name(reduce_max_region["def_out"])) == {"scores_max"}
    assert set(buffer_regions_by_name(reduce_sum_region["use_in"])) == {"acc_s", "scores_sum"}
    assert set(buffer_regions_by_name(reduce_sum_region["def_out"])) == {"scores_sum"}

    tile_outer_scopes = find_scope_nodes_with_extents(plan["tree"], ["i"], ["8"])
    row_outer_scopes = find_scope_nodes_with_extents(plan["tree"], ["i"], ["1"])
    inner_scope = find_scope_nodes_with_extents(plan["tree"], ["i", "j"], ["8", "1"])

    assert find_scope_nodes_with_extents(plan["tree"], ["i"], ["32"]) == []
    assert len(tile_outer_scopes) == 1
    assert len(row_outer_scopes) == 1
    assert len(inner_scope) == 1
    assert child_region_indices(tile_outer_scopes[0]) == []
    assert child_region_indices(row_outer_scopes[0]) == [3, 4]
    assert child_region_indices(inner_scope[0]) == [5, 8, 6]
    assert [node["region_index"] for node in plan["tree"] if not node["is_scope"]] == [0, 1, 2, 7, 9]


def test_sunmmio_tile_loop_fusion_planner_attention_sink_keeps_late_row_update_under_outer_shell():
    mod = IRModule.from_expr(attention_sink_online_softmax_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tiles_lowering(mod)

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    plan_summary = get_plan_summary(mod)
    (graph,) = dependence["graphs"]
    (plan,) = plan_summary["plans"]

    assert discovery["region_count"] == 11
    assert discovery["region_run_count"] == 1
    assert discovery["region_run_lengths"] == [11]
    assert plan_summary["plan_count"] == 1
    assert {
        (5, 8, "RAW", "acc_s", 2, 1024),
        (6, 7, "RAW", "scores_sum", 1, 32),
        (7, 10, "RAW", "logsum", 1, 256),
        (7, 10, "WAR", "logsum", 1, 0),
        (7, 10, "WAW", "logsum", 1, 0),
        (3, 10, "RAW", "scores_max", 1, 256),
        (4, 9, "RAW", "scores_scale", 1, 32),
    } <= graph_edge_keys(graph)
    assert plan["region_indices"] == list(range(11))
    assert plan["order"] == [0, 1, 2, 3, 5, 8, 6, 4, 7, 10, 9]
    assert plan["score"] == {
        "write_cut_cost": 2048,
        "shared_read_cost": 12704,
        "live_range_penalty": 26880,
        "reorder_penalty": 6,
    }

    tile_outer_scopes = find_scope_nodes_with_extents(plan["tree"], ["i"], ["8"])
    row_outer_scopes = find_scope_nodes_with_extents(plan["tree"], ["i"], ["1"])
    inner_scope = find_scope_nodes_with_extents(plan["tree"], ["i", "j"], ["8", "1"])

    assert len(tile_outer_scopes) == 1
    assert len(row_outer_scopes) == 1
    assert len(inner_scope) == 1
    assert child_region_indices(tile_outer_scopes[0]) == []
    assert child_region_indices(row_outer_scopes[0]) == [4, 7, 10]
    assert child_region_indices(inner_scope[0]) == [5, 8, 6]
    assert [node["region_index"] for node in plan["tree"] if not node["is_scope"]] == [0, 1, 2, 3, 9]


def test_sunmmio_tile_loop_fusion_planner_nsa_chain_reorders_without_row_merge_loop():
    mod = IRModule.from_expr(nsa_forward_online_softmax_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tiles_lowering(mod)

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    plan_summary = get_plan_summary(mod)
    (graph,) = dependence["graphs"]
    (plan,) = plan_summary["plans"]

    assert discovery["region_count"] == 9
    assert discovery["region_run_count"] == 1
    assert discovery["region_run_lengths"] == [9]
    assert plan_summary["plan_count"] == 1
    assert {
        (0, 1, "WAR", "scores_max", 1, 0),
        (0, 3, "RAW", "scores_max_prev", 1, 256),
        (2, 4, "RAW", "scores_max", 1, 32),
        (4, 5, "RAW", "acc_s", 2, 1024),
        (5, 6, "RAW", "scores_sum", 1, 32),
        (4, 7, "RAW", "acc_s", 2, 1024),
        (3, 8, "RAW", "scores_scale", 1, 32),
    } <= graph_edge_keys(graph)
    assert plan["region_indices"] == list(range(9))
    assert plan["order"] == [0, 1, 2, 4, 7, 5, 3, 6, 8]
    assert plan["score"] == {
        "write_cut_cost": 1088,
        "shared_read_cost": 8480,
        "live_range_penalty": 31232,
        "reorder_penalty": 5,
    }

    tile_outer_scopes = find_scope_nodes_with_extents(plan["tree"], ["i"], ["8"])
    row_outer_scopes = find_scope_nodes_with_extents(plan["tree"], ["i"], ["1"])
    inner_scope = find_scope_nodes_with_extents(plan["tree"], ["i", "j"], ["8", "1"])

    assert len(tile_outer_scopes) == 1
    assert len(row_outer_scopes) == 1
    assert len(inner_scope) == 1
    assert child_region_indices(tile_outer_scopes[0]) == []
    assert child_region_indices(row_outer_scopes[0]) == [3, 6]
    assert child_region_indices(inner_scope[0]) == [2, 4, 7, 5]
    assert [node["region_index"] for node in plan["tree"] if not node["is_scope"]] == [0, 1, 8]


def test_sunmmio_tile_loop_fusion_planner_rms_norm_explicit_reduce_forms_expected_outer_and_inner_shells():
    mod = IRModule.from_expr(rms_norm_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tiles_lowering(mod)

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    plan_summary = get_plan_summary(mod)
    (graph,) = dependence["graphs"]
    (plan,) = plan_summary["plans"]

    assert discovery["region_count"] == 5
    assert discovery["region_run_count"] == 1
    assert discovery["region_run_lengths"] == [5]
    assert plan_summary["plan_count"] == 1
    assert graph_edge_keys(graph) == {
        (0, 2, "RAW", "row_sum", 1, 32),
        (1, 2, "RAW", "a_square", 2, 1024),
        (0, 2, "WAW", "row_sum", 1, 0),
        (2, 3, "RAW", "row_sum", 1, 32),
        (3, 4, "RAW", "row_scale", 1, 32),
    }
    assert plan["region_indices"] == [0, 1, 2, 3, 4]
    assert plan["order"] == [0, 1, 2, 3, 4]
    assert plan["score"] == {
        "write_cut_cost": 768,
        "shared_read_cost": 8192,
        "live_range_penalty": 20736,
        "reorder_penalty": 0,
    }
    assert [region["execution_loop_extents"] for region in discovery["regions"]] == [
        ["1"],
        ["8", "1"],
        ["8", "1"],
        ["1"],
        ["8", "1"],
    ]
    rms_outer_scopes = find_scope_nodes_with_extents(plan["tree"], ["i"], ["8"])
    rms_inner_scopes = find_scope_nodes_with_extents(plan["tree"], ["i", "j"], ["8", "1"])

    assert len(rms_outer_scopes) == 1
    assert len(rms_inner_scopes) == 1
    assert child_region_indices(rms_outer_scopes[0]) == []
    assert child_region_indices(rms_inner_scopes[0]) == [1, 2]
    assert [node["region_index"] for node in plan["tree"] if not node["is_scope"]] == [0, 3, 4]


def test_sunmmio_tile_loop_fusion_planner_rms_norm_fill_and_rsram_copy_init_surface_init_ops_and_form_row_shells():
    fill_mod = IRModule.from_expr(rms_norm_tiled_fill_init_kernel().with_attr("global_symbol", "main"))
    copy_mod = IRModule.from_expr(rms_norm_tiled_rsram_copy_init_kernel().with_attr("global_symbol", "main"))
    fill_mod = apply_sunmmio_tiles_lowering(fill_mod)
    copy_mod = apply_sunmmio_tiles_lowering(copy_mod)

    fill_discovery = get_discovery_summary(fill_mod)
    fill_dependence = get_dependence_summary(fill_mod)
    fill_plan_summary = get_plan_summary(fill_mod)
    copy_discovery = get_discovery_summary(copy_mod)
    copy_dependence = get_dependence_summary(copy_mod)
    copy_plan_summary = get_plan_summary(copy_mod)

    (fill_graph,) = fill_dependence["graphs"]
    (fill_plan,) = fill_plan_summary["plans"]
    (copy_graph,) = copy_dependence["graphs"]
    (copy_plan,) = copy_plan_summary["plans"]

    assert fill_discovery["region_count"] == 5
    assert fill_discovery["region_run_count"] == 1
    assert fill_discovery["region_run_lengths"] == [5]
    assert fill_plan_summary["plan_count"] == 1
    assert graph_edge_keys(fill_graph) == {
        (1, 2, "RAW", "a_square", 2, 1024),
        (0, 2, "RAW", "row_sum", 1, 32),
        (0, 2, "WAW", "row_sum", 1, 0),
        (2, 3, "RAW", "row_sum", 1, 32),
        (3, 4, "RAW", "row_scale", 1, 32),
    }
    assert fill_plan["order"] == [0, 1, 2, 3, 4]
    assert fill_plan["score"] == {
        "write_cut_cost": 768,
        "shared_read_cost": 8192,
        "live_range_penalty": 20736,
        "reorder_penalty": 0,
    }

    assert copy_discovery["region_count"] == 6
    assert copy_discovery["region_run_count"] == 1
    assert copy_discovery["region_run_lengths"] == [6]
    assert copy_plan_summary["plan_count"] == 1
    assert graph_edge_keys(copy_graph) == {
        (0, 1, "RAW", "row_zero", 1, 256),
        (2, 3, "RAW", "a_square", 2, 1024),
        (1, 3, "RAW", "row_sum", 1, 32),
        (1, 3, "WAW", "row_sum", 1, 0),
        (3, 4, "RAW", "row_sum", 1, 32),
        (4, 5, "RAW", "row_scale", 1, 32),
    }
    assert copy_plan["order"] == [0, 1, 2, 3, 4, 5]
    assert copy_plan["score"] == {
        "write_cut_cost": 768,
        "shared_read_cost": 8192,
        "live_range_penalty": 21376,
        "reorder_penalty": 0,
    }

    fill_row_sum_regions = [region for region in fill_discovery["regions"] if "row_sum" in buffer_regions_by_name(region["def_out"])]
    copy_row_sum_regions = [region for region in copy_discovery["regions"] if "row_sum" in buffer_regions_by_name(region["def_out"])]
    copy_row_zero_regions = [region for region in copy_discovery["regions"] if "row_zero" in buffer_regions_by_name(region["def_out"])]

    fill_outer_scopes = find_scope_nodes_with_extents(fill_plan["tree"], ["i"], ["8"])
    fill_inner_scopes = find_scope_nodes_with_extents(fill_plan["tree"], ["i", "j"], ["8", "1"])
    copy_outer_scopes = find_scope_nodes_with_extents(copy_plan["tree"], ["i"], ["1"])
    copy_tile_outer_scopes = find_scope_nodes_with_extents(copy_plan["tree"], ["i"], ["8"])
    copy_inner_scopes = find_scope_nodes_with_extents(copy_plan["tree"], ["i", "j"], ["8", "1"])

    assert len(fill_outer_scopes) == 1
    assert len(fill_inner_scopes) == 1
    assert child_region_indices(fill_outer_scopes[0]) == []
    assert child_region_indices(fill_inner_scopes[0]) == [1, 2]

    assert len(copy_outer_scopes) == 1
    assert child_region_indices(copy_outer_scopes[0]) == [0, 1]
    assert len(copy_tile_outer_scopes) == 1
    assert len(copy_inner_scopes) == 1
    assert child_region_indices(copy_tile_outer_scopes[0]) == []
    assert child_region_indices(copy_inner_scopes[0]) == [2, 3]

    assert len(fill_row_sum_regions) == 2
    assert len(copy_row_sum_regions) == 2
    assert len(copy_row_zero_regions) == 1
