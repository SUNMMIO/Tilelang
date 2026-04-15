from testing.python.transform._sunmmio_tile_loop_fusion_test_utils import *


def test_sunmmio_tile_loop_fusion_analysis_single_region_single_window():
    mod = IRModule.from_expr(single_tile_scope_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    summary = get_debug_summary(mod)
    region = summary["regions"][0]
    use_in = buffer_regions_by_name(region["use_in"])
    def_out = buffer_regions_by_name(region["def_out"])

    assert summary["region_count"] == 1
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [1]
    assert summary["graphs"] == [{"region_indices": [0], "edges": []}]
    assert summary["plans"] == [
        {
            "region_indices": [0],
            "order": [0],
            "score": {
                "write_cut_cost": 0,
                "shared_read_cost": 2048,
                "live_range_penalty": 0,
                "reorder_penalty": 0,
            },
            "actions": [
                {
                    "region_index": 0,
                    "close_to_depth": 0,
                    "open_to_depth": 0,
                    "opened_shells": [],
                    "opened_shell_extents": [],
                }
            ],
            "tree": [
                {
                    "is_scope": False,
                    "region_index": 0,
                    "shell_axes": [],
                    "shell_extents": [],
                    "children": [],
                }
            ],
        }
    ]
    assert region["root_loop_var"] == "i"
    assert region["execution_loop_vars"] == ["i", "j"]
    assert region["sig_rank"] == 2
    assert region["sig_tile_shape"] == [8, 32]
    assert region["sig_execution_axis_to_loop_index"] == [0, 1]
    assert set(use_in) == {"A_shared"}
    assert use_in["A_shared"] == {
        "buffer": "A_shared",
        "mins": ["i * 8", "j * 32"],
        "extents": ["8", "32"],
    }
    assert set(def_out) == {"B_shared"}
    assert def_out["B_shared"] == {
        "buffer": "B_shared",
        "mins": ["i * 8", "j * 32"],
        "extents": ["8", "32"],
    }
    assert region["available_at_execution_depths"] == [2]


def test_sunmmio_tile_loop_fusion_analysis_two_consecutive_regions_one_window():
    mod = IRModule.from_expr(two_consecutive_tile_scopes_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    summary = get_debug_summary(mod)
    first, second = summary["regions"]
    graph = summary["graphs"][0]
    plan = summary["plans"][0]

    assert summary["region_count"] == 2
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [2]
    assert graph["region_indices"] == [0, 1]
    assert graph["edges"] == [
        {
            "src": 0,
            "dst": 1,
            "kind": "RAW",
            "buffer_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "w": 1024,
        }
    ]
    assert plan == {
        "region_indices": [0, 1],
        "order": [0, 1],
        "score": {
            "write_cut_cost": 0,
            "shared_read_cost": 2048,
            "live_range_penalty": 6144,
            "reorder_penalty": 0,
        },
        "actions": [
            {
                "region_index": 0,
                "close_to_depth": 0,
                "open_to_depth": 2,
                "opened_shells": [["i"], ["i", "j"]],
                "opened_shell_extents": [["4"], ["4", "1"]],
            },
            {
                "region_index": 1,
                "close_to_depth": 2,
                "open_to_depth": 2,
                "opened_shells": [],
                "opened_shell_extents": [],
            },
        ],
        "tree": [
            {
                "is_scope": True,
                "region_index": -1,
                "shell_axes": ["i"],
                "shell_extents": ["4"],
                "children": [
                    {
                        "is_scope": True,
                        "region_index": -1,
                        "shell_axes": ["i", "j"],
                        "shell_extents": ["4", "1"],
                        "children": [
                            {
                                "is_scope": False,
                                "region_index": 0,
                                "shell_axes": [],
                                "shell_extents": [],
                                "children": [],
                            },
                            {
                                "is_scope": False,
                                "region_index": 1,
                                "shell_axes": [],
                                "shell_extents": [],
                                "children": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }
    assert buffer_regions_by_name(first["use_in"]) == {
        "A_shared": {
            "buffer": "A_shared",
            "mins": ["i * 8", "j * 32"],
            "extents": ["8", "32"],
        }
    }
    assert buffer_regions_by_name(first["def_out"]) == {
        "Tmp_shared": {
            "buffer": "Tmp_shared",
            "mins": ["i * 8", "j * 32"],
            "extents": ["8", "32"],
        }
    }
    assert first["available_at_execution_depths"] == [2]
    assert buffer_regions_by_name(second["use_in"]) == {
        "Tmp_shared": {
            "buffer": "Tmp_shared",
            "mins": ["i * 8", "j * 32"],
            "extents": ["8", "32"],
        }
    }
    assert buffer_regions_by_name(second["def_out"]) == {
        "B_shared": {
            "buffer": "B_shared",
            "mins": ["i * 8", "j * 32"],
            "extents": ["8", "32"],
        }
    }
    assert second["available_at_execution_depths"] == [2]


def test_sunmmio_tile_loop_fusion_legality_graph_kills_overwritten_defs():
    mod = IRModule.from_expr(overwritten_tile_scope_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    summary = get_debug_summary(mod)
    graph = summary["graphs"][0]

    assert summary["region_count"] == 3
    assert summary["window_lengths"] == [3]
    assert graph["region_indices"] == [0, 1, 2]
    assert graph["edges"] == [
        {
            "src": 0,
            "dst": 1,
            "kind": "WAW",
            "buffer_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "w": 0,
        },
        {
            "src": 1,
            "dst": 2,
            "kind": "RAW",
            "buffer_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "w": 1024,
        },
    ]


def test_sunmmio_tile_loop_fusion_legality_graph_records_war_edges():
    mod = IRModule.from_expr(read_then_overwrite_tile_scope_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    summary = get_debug_summary(mod)
    graph = summary["graphs"][0]

    assert summary["region_count"] == 2
    assert summary["window_lengths"] == [2]
    assert graph["region_indices"] == [0, 1]
    assert graph["edges"] == [
        {
            "src": 0,
            "dst": 1,
            "kind": "WAR",
            "buffer_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "w": 0,
        }
    ]


def test_sunmmio_tile_loop_fusion_analysis_3d_region_restricts_to_exposed_prefix():
    mod = IRModule.from_expr(
        dot_mul_tiled_parallel_3d(
            Batch=64,
            M=512,
            N=1024,
            block_B=16,
            block_M=256,
            block_N=128,
            tile_size=(2, 128),
            index_map=(-2, -1),
        ).with_attr("global_symbol", "main")
    )
    mod = apply_tiles_lowering(mod)

    summary = get_debug_summary(mod)
    region = summary["regions"][0]
    use_in = buffer_regions_by_name(region["use_in"])

    assert summary["region_count"] == 1
    assert region["root_loop_var"] == "i"
    assert region["execution_loop_vars"] == ["i", "j"]
    assert region["sig_rank"] == 2
    assert region["sig_tile_shape"] == [2, 128]
    assert region["sig_execution_axis_to_loop_index"] == [0, 1]
    assert set(use_in) == {"A_shared", "B_shared"}
    assert use_in["A_shared"] == {
        "buffer": "A_shared",
        "mins": ["b", "i * 2", "j * 128"],
        "extents": ["1", "2", "128"],
    }
    assert region["available_at_execution_depths"] == []


def test_sunmmio_tile_loop_fusion_analysis_swapped_domain_exec_map():
    mod = IRModule.from_expr(dot_mul_tiled_parallel_2d_swapped_domain(M=256, N=128).with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    summary = get_debug_summary(mod)
    region = summary["regions"][0]

    assert summary["region_count"] == 1
    assert region["root_loop_var"] == "j"
    assert region["execution_loop_vars"] == ["j", "i"]
    assert region["sig_rank"] == 2
    assert region["sig_tile_shape"] == [2, 128]
    assert region["sig_execution_axis_to_loop_index"] == [1, 0]
    assert region["available_at_execution_depths"] == [2]
