from testing.python.transform._sunmmio_tile_loop_fusion_test_utils import *


def test_sunmmio_tile_loop_fusion_analysis_single_region_single_window():
    mod = IRModule.from_expr(single_tile_scope_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    region = discovery["regions"][0]
    use_in = buffer_regions_by_name(region["use_in"])
    def_out = buffer_regions_by_name(region["def_out"])

    assert discovery["region_count"] == 1
    assert discovery["region_run_count"] == 1
    assert discovery["region_run_lengths"] == [1]
    assert dependence["graphs"] == [{"region_indices": [0], "edges": []}]
    assert region["root_loop_var"] == "i"
    assert region["execution_loop_vars"] == ["i", "j"]
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

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    first, second = discovery["regions"]
    graph = dependence["graphs"][0]

    assert discovery["region_count"] == 2
    assert discovery["region_run_count"] == 1
    assert discovery["region_run_lengths"] == [2]
    assert graph["region_indices"] == [0, 1]
    assert graph["edges"] == [
        {
            "src": 0,
            "dst": 1,
            "kind": "RAW",
            "src_access_index": 0,
            "dst_access_index": 0,
            "buffer": "Tmp_shared",
            "debug_overlap_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "weight_bytes": 1024,
        }
    ]
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

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    graph = dependence["graphs"][0]

    assert discovery["region_count"] == 3
    assert discovery["region_run_lengths"] == [3]
    assert graph["region_indices"] == [0, 1, 2]
    assert graph["edges"] == [
        {
            "src": 0,
            "dst": 1,
            "kind": "WAW",
            "src_access_index": 0,
            "dst_access_index": 0,
            "buffer": "Tmp_shared",
            "debug_overlap_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "weight_bytes": 0,
        },
        {
            "src": 1,
            "dst": 2,
            "kind": "RAW",
            "src_access_index": 0,
            "dst_access_index": 0,
            "buffer": "Tmp_shared",
            "debug_overlap_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "weight_bytes": 1024,
        },
    ]


def test_sunmmio_tile_loop_fusion_legality_graph_records_war_edges():
    mod = IRModule.from_expr(read_then_overwrite_tile_scope_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    discovery = get_discovery_summary(mod)
    dependence = get_dependence_summary(mod)
    graph = dependence["graphs"][0]

    assert discovery["region_count"] == 2
    assert discovery["region_run_lengths"] == [2]
    assert graph["region_indices"] == [0, 1]
    assert graph["edges"] == [
        {
            "src": 0,
            "dst": 1,
            "kind": "WAR",
            "src_access_index": 0,
            "dst_access_index": 0,
            "buffer": "Tmp_shared",
            "debug_overlap_region": {
                "buffer": "Tmp_shared",
                "mins": ["i * 8", "j * 32"],
                "extents": ["8", "32"],
            },
            "rho": 2,
            "weight_bytes": 0,
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

    discovery = get_discovery_summary(mod)
    region = discovery["regions"][0]
    use_in = buffer_regions_by_name(region["use_in"])

    assert discovery["region_count"] == 1
    assert region["root_loop_var"] == "i"
    assert region["execution_loop_vars"] == ["i", "j"]
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

    discovery = get_discovery_summary(mod)
    region = discovery["regions"][0]

    assert discovery["region_count"] == 1
    assert region["root_loop_var"] == "j"
    assert region["execution_loop_vars"] == ["j", "i"]
    assert region["available_at_execution_depths"] == [2]
