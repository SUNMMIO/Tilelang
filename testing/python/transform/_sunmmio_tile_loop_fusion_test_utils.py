import linecache


import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import make_blockwise_zz_layout
from tilelang.tileview import make_tileview
from tilelang.utils.target import SUNMMIO_TARGET_DESC


def apply_tiles_lowering(mod):
    return tl.transform.LowerTilesLoop()(mod)


def apply_sunmmio_tiles_lowering(mod):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tl.transform.AddWrapperForSingleBufStore()(mod)
        mod = tl.transform.LegalizeNegativeIndex()(mod)
        mod = tl.transform.InjectAssumes()(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.transform.InferSramScope()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
        mod = tl.transform.LowerTilesLoop()(mod)
    return mod


def apply_sunmmio_tile_loop_fusion(mod):
    mod = apply_sunmmio_tiles_lowering(mod)
    return tl.transform.SunmmioTileLoopFusion()(mod)


def _to_buffer_region_dict(region):
    return {
        "buffer": str(region["buffer"]),
        "mins": [str(x) for x in region["mins"]],
        "extents": [str(x) for x in region["extents"]],
    }


def _to_region_run_dict(region_run):
    return {
        "begin_region_index": int(region_run["begin_region_index"]),
        "num_regions": int(region_run["num_regions"]),
    }


def _to_region_dict(region):
    return {
        "root_loop_var": str(region["root_loop_var"]),
        "execution_loop_vars": [str(x) for x in region["execution_loop_vars"]],
        "logical_execution_axes": [str(x) for x in region["logical_execution_axes"]],
        "execution_loop_extents": [str(x) for x in region["execution_loop_extents"]],
        "use_in": [_to_buffer_region_dict(x) for x in region["use_in"]],
        "def_out": [_to_buffer_region_dict(x) for x in region["def_out"]],
        "available_at_execution_depths": [int(x) for x in region["available_at_execution_depths"]],
    }


def _to_edge_dict(edge):
    return {
        "src": int(edge["src"]),
        "dst": int(edge["dst"]),
        "kind": str(edge["kind"]),
        "src_access_index": int(edge["src_access_index"]),
        "dst_access_index": int(edge["dst_access_index"]),
        "buffer": str(edge["buffer"]),
        "debug_overlap_region": _to_buffer_region_dict(edge["debug_overlap_region"]),
        "rho": int(edge["rho"]),
        "weight_bytes": int(edge["weight_bytes"]),
    }


def _to_graph_dict(graph):
    return {
        "region_indices": [int(x) for x in graph["region_indices"]],
        "edges": [_to_edge_dict(x) for x in graph["edges"]],
    }


def _to_bool(value):
    text = str(value)
    if text in ("True", "T.bool(True)"):
        return True
    if text in ("False", "T.bool(False)"):
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _to_score_dict(score):
    return {
        "write_cut_cost": int(score["write_cut_cost"]),
        "shared_read_cost": int(score["shared_read_cost"]),
        "live_range_penalty": int(score["live_range_penalty"]),
        "reorder_penalty": int(score["reorder_penalty"]),
    }


def _to_action_dict(action):
    return {
        "region_index": int(action["region_index"]),
        "close_to_depth": int(action["close_to_depth"]),
        "open_to_depth": int(action["open_to_depth"]),
        "opened_shells": [[str(axis) for axis in shell] for shell in action["opened_shells"]],
        "opened_shell_extents": [[str(extent) for extent in shell] for shell in action["opened_shell_extents"]],
    }


def _to_tree_node_dict(node):
    return {
        "is_scope": _to_bool(node["is_scope"]),
        "region_index": int(node["region_index"]),
        "shell_axes": [str(axis) for axis in node["shell_axes"]],
        "shell_extents": [str(extent) for extent in node["shell_extents"]],
        "children": [_to_tree_node_dict(child) for child in node["children"]],
    }


def _to_plan_dict(plan):
    return {
        "region_indices": [int(x) for x in plan["region_indices"]],
        "order": [int(x) for x in plan["order"]],
        "score": _to_score_dict(plan["score"]),
        "actions": [_to_action_dict(x) for x in plan["actions"]],
        "tree": [_to_tree_node_dict(x) for x in plan["tree"]],
    }


def get_discovery_summary(mod):
    explain_func = tvm.get_global_func("tl.analysis.ExplainSunmmioTileLoopFusionDiscovery")
    summary = explain_func(mod)
    return {
        "region_count": int(summary["region_count"]),
        "region_run_count": int(summary["region_run_count"]),
        "region_run_lengths": [int(x) for x in summary["region_run_lengths"]],
        "region_runs": [_to_region_run_dict(x) for x in summary["region_runs"]],
        "regions": [_to_region_dict(x) for x in summary["regions"]],
    }


def get_dependence_summary(mod):
    explain_func = tvm.get_global_func("tl.analysis.ExplainSunmmioTileLoopFusionDependence")
    summary = explain_func(mod)
    return {
        "region_run_count": int(summary["region_run_count"]),
        "region_run_lengths": [int(x) for x in summary["region_run_lengths"]],
        "graphs": [_to_graph_dict(x) for x in summary["graphs"]],
    }


def get_plan_summary(mod):
    explain_func = tvm.get_global_func("tl.analysis.ExplainSunmmioTileLoopFusionPlan")
    summary = explain_func(mod)
    return {
        "plan_count": int(summary["plan_count"]),
        "plans": [_to_plan_dict(x) for x in summary["plans"]],
    }


def get_phase1_guardrail_summary():
    debug_func = tvm.get_global_func("tl.analysis.DebugSunmmioTileLoopFusionPhase1Guardrails")
    summary = debug_func()
    return {
        "planner_cost_limit": int(summary["planner_cost_limit"]),
        "saturating_add_overflow": int(summary["saturating_add_overflow"]),
        "saturating_mul_overflow": int(summary["saturating_mul_overflow"]),
        "resident_dedupe_identical_count": int(summary["resident_dedupe_identical_count"]),
        "resident_dedupe_payload_distinct_count": int(summary["resident_dedupe_payload_distinct_count"]),
        "resident_dedupe_instance_distinct_count": int(summary["resident_dedupe_instance_distinct_count"]),
    }


def check_phase1_bitset_bounds(num_bits, index):
    debug_func = tvm.get_global_func("tl.analysis.DebugSunmmioTileLoopFusionCheckBitsetBounds")
    return bool(debug_func(num_bits, index))


def get_raw_coverage_accounting_summary():
    debug_func = tvm.get_global_func("tl.analysis.DebugSunmmioTileLoopFusionRawCoverageAccounting")
    summary = debug_func()
    return {
        "first_write_cut_cost": int(summary["first_write_cut_cost"]),
        "first_shared_read_cost": int(summary["first_shared_read_cost"]),
        "second_write_cut_cost": int(summary["second_write_cut_cost"]),
        "second_shared_read_cost": int(summary["second_shared_read_cost"]),
    }


def buffer_regions_by_name(regions):
    return {region["buffer"]: region for region in regions}


def walk_plan_tree(nodes):
    for node in nodes:
        yield node
        yield from walk_plan_tree(node["children"])


def find_scope_nodes(nodes, shell_axes):
    return [node for node in walk_plan_tree(nodes) if node["is_scope"] and node["shell_axes"] == shell_axes]


def find_scope_nodes_with_extents(nodes, shell_axes, shell_extents):
    return [
        node
        for node in walk_plan_tree(nodes)
        if node["is_scope"] and node["shell_axes"] == shell_axes and node["shell_extents"] == shell_extents
    ]


def child_region_indices(node):
    return [child["region_index"] for child in node["children"] if not child["is_scope"]]


def graph_edge_keys(graph):
    return {
        (
            edge["src"],
            edge["dst"],
            edge["kind"],
            edge["buffer"],
            edge["rho"],
            edge["weight_bytes"],
        )
        for edge in graph["edges"]
    }


def single_tile_scope_kernel(block_m=32, block_n=32, tile_size=(8, 32), dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((block_m, block_n), dtype), B: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, (-2, -1)),
                    B_shared: make_tileview(B_shared, tile_size, (-2, -1)),
                }
            )

            T.copy(A[0:block_m, 0:block_n], A_shared)
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                B_shared[i, j] = A_shared[i, j]
            T.copy(B_shared, B[0:block_m, 0:block_n])

    return main


def two_consecutive_tile_scopes_kernel(block_m=32, block_n=32, tile_size=(8, 32), dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((block_m, block_n), dtype), B: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            Tmp_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    Tmp_shared: make_blockwise_zz_layout(Tmp_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, (-2, -1)),
                    Tmp_shared: make_tileview(Tmp_shared, tile_size, (-2, -1)),
                    B_shared: make_tileview(B_shared, tile_size, (-2, -1)),
                }
            )

            T.copy(A[0:block_m, 0:block_n], A_shared)
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                Tmp_shared[i, j] = A_shared[i, j]
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                B_shared[i, j] = Tmp_shared[i, j]
            T.copy(B_shared, B[0:block_m, 0:block_n])

    return main


def independent_tile_scopes_kernel(block_m=32, block_n=32, tile_size=(8, 32), dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((block_m, block_n), dtype),
        B: T.Tensor((block_m, block_n), dtype),
        C: T.Tensor((block_m, block_n), dtype),
        D: T.Tensor((block_m, block_n), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_m, block_n), dtype)
            D_shared = T.alloc_shared((block_m, block_n), dtype)

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                    D_shared: make_blockwise_zz_layout(D_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, (-2, -1)),
                    B_shared: make_tileview(B_shared, tile_size, (-2, -1)),
                    C_shared: make_tileview(C_shared, tile_size, (-2, -1)),
                    D_shared: make_tileview(D_shared, tile_size, (-2, -1)),
                }
            )

            T.copy(A[0:block_m, 0:block_n], A_shared)
            T.copy(B[0:block_m, 0:block_n], B_shared)
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                C_shared[i, j] = A_shared[i, j]
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                D_shared[i, j] = B_shared[i, j]
            T.copy(C_shared, C[0:block_m, 0:block_n])
            T.copy(D_shared, D[0:block_m, 0:block_n])

    return main


def flash_attention_online_softmax_tiled_kernel(block_m=32, block_n=32, dim=32, dtype="float32", accum_dtype="float32"):
    scale = 1.0

    @T.prim_func
    def main(
        AccSIn: T.Tensor((block_m, block_n), accum_dtype),
        AccOIn: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxIn: T.Tensor((block_m,), accum_dtype),
        LogsumIn: T.Tensor((block_m,), accum_dtype),
        AccSCastOut: T.Tensor((block_m, block_n), accum_dtype),
        AccOOut: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxOut: T.Tensor((block_m,), accum_dtype),
        LogsumOut: T.Tensor((block_m,), accum_dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            acc_s = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_s_cast = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_o = T.alloc_shared((block_m, dim), accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_max_prev = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            logsum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AccSIn[0:block_m, 0:block_n], acc_s)
            T.copy(AccOIn[0:block_m, 0:dim], acc_o)
            T.copy(ScoresMaxIn[0:block_m], scores_max)
            T.copy(LogsumIn[0:block_m], logsum)

            for i in T.Tiles([block_m], parallel=True):
                scores_max_prev[i] = scores_max[i]

            for i in T.Tiles([block_m], parallel=True):
                scores_max[i] = -T.infinity(accum_dtype)

            T.reduce(acc_s, scores_max, "max", dim=1, clear=False)

            for i in T.Tiles([block_m], parallel=True):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

            for i in T.Tiles([block_m], parallel=True):
                scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(scale) - scores_max[i] * T.float32(scale))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(scale) - scores_max[i] * T.float32(scale))

            T.reduce(acc_s, scores_sum, "sum", dim=1, clear=True)

            for i in T.Tiles([block_m], parallel=True):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s_cast[i, j] = acc_s[i, j]

            for i, j in T.Tiles([block_m, dim], parallel=True):
                acc_o[i, j] *= scores_scale[i]

            T.copy(acc_s_cast, AccSCastOut[0:block_m, 0:block_n])
            T.copy(acc_o, AccOOut[0:block_m, 0:dim])
            T.copy(scores_max, ScoresMaxOut[0:block_m])
            T.copy(logsum, LogsumOut[0:block_m])

    return main


def attention_sink_online_softmax_tiled_kernel(block_m=32, block_n=32, dim=32, dtype="float32", accum_dtype="float32"):
    scale = 1.0

    @T.prim_func
    def main(
        AccSIn: T.Tensor((block_m, block_n), accum_dtype),
        AccOIn: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxIn: T.Tensor((block_m,), accum_dtype),
        LogsumIn: T.Tensor((block_m,), accum_dtype),
        SinkIn: T.Tensor((block_m,), accum_dtype),
        AccSCastOut: T.Tensor((block_m, block_n), accum_dtype),
        AccOOut: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxOut: T.Tensor((block_m,), accum_dtype),
        LogsumOut: T.Tensor((block_m,), accum_dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            acc_s = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_s_cast = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_o = T.alloc_shared((block_m, dim), accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_max_prev = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            logsum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            sink = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AccSIn[0:block_m, 0:block_n], acc_s)
            T.copy(AccOIn[0:block_m, 0:dim], acc_o)
            T.copy(ScoresMaxIn[0:block_m], scores_max)
            T.copy(LogsumIn[0:block_m], logsum)
            T.copy(SinkIn[0:block_m], sink)

            for i in T.Tiles([block_m], parallel=True):
                scores_max_prev[i] = scores_max[i]

            for i in T.Tiles([block_m], parallel=True):
                scores_max[i] = -T.infinity(accum_dtype)

            T.reduce(acc_s, scores_max, "max", dim=1, clear=False)

            for i in T.Tiles([block_m], parallel=True):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

            for i in T.Tiles([block_m], parallel=True):
                scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(scale) - scores_max[i] * T.float32(scale))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(scale) - scores_max[i] * T.float32(scale))

            T.reduce(acc_s, scores_sum, "sum", dim=1, clear=True)

            for i in T.Tiles([block_m], parallel=True):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s_cast[i, j] = acc_s[i, j]

            for i, j in T.Tiles([block_m, dim], parallel=True):
                acc_o[i, j] *= scores_scale[i]

            for i in T.Tiles([block_m], parallel=True):
                logsum[i] = logsum[i] + T.exp2(sink[i] - scores_max[i] * T.float32(scale))

            T.copy(acc_s_cast, AccSCastOut[0:block_m, 0:block_n])
            T.copy(acc_o, AccOOut[0:block_m, 0:dim])
            T.copy(scores_max, ScoresMaxOut[0:block_m])
            T.copy(logsum, LogsumOut[0:block_m])

    return main


def nsa_forward_online_softmax_tiled_kernel(block_m=32, block_n=32, dim=32, dtype="float32", accum_dtype="float32"):
    scale = 1.0

    @T.prim_func
    def main(
        AccSIn: T.Tensor((block_m, block_n), accum_dtype),
        AccOIn: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxIn: T.Tensor((block_m,), accum_dtype),
        LogsumIn: T.Tensor((block_m,), accum_dtype),
        AccSCastOut: T.Tensor((block_m, block_n), accum_dtype),
        AccOOut: T.Tensor((block_m, dim), accum_dtype),
        ScoresMaxOut: T.Tensor((block_m,), accum_dtype),
        LogsumOut: T.Tensor((block_m,), accum_dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            acc_s = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_s_cast = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            acc_o = T.alloc_shared((block_m, dim), accum_dtype, scope="shared.rsram")
            scores_max = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_max_prev = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            scores_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            logsum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AccSIn[0:block_m, 0:block_n], acc_s)
            T.copy(AccOIn[0:block_m, 0:dim], acc_o)
            T.copy(ScoresMaxIn[0:block_m], scores_max)
            T.copy(LogsumIn[0:block_m], logsum)

            for i in T.Tiles([block_m], parallel=True):
                scores_max_prev[i] = scores_max[i]

            for i in T.Tiles([block_m], parallel=True):
                scores_max[i] = -T.infinity(accum_dtype)

            T.reduce(acc_s, scores_max, "max", dim=1, clear=True)

            for i in T.Tiles([block_m], parallel=True):
                scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(scale) - scores_max[i] * T.float32(scale))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(scale) - scores_max[i] * T.float32(scale))

            T.reduce(acc_s, scores_sum, "sum", dim=1, clear=True)

            for i in T.Tiles([block_m], parallel=True):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                acc_s_cast[i, j] = acc_s[i, j]

            for i, j in T.Tiles([block_m, dim], parallel=True):
                acc_o[i, j] *= scores_scale[i]

            T.copy(acc_s_cast, AccSCastOut[0:block_m, 0:block_n])
            T.copy(acc_o, AccOOut[0:block_m, 0:dim])
            T.copy(scores_max, ScoresMaxOut[0:block_m])
            T.copy(logsum, LogsumOut[0:block_m])

    return main


def rms_norm_tiled_kernel(block_m=32, block_n=32, dtype="float32", accum_dtype="float32"):
    eps = 1e-6

    @T.prim_func
    def main(AIn: T.Tensor((block_m, block_n), dtype), AOut: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            a_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            a_square = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            a_out = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            row_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            row_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AIn[0:block_m, 0:block_n], a_shared)

            for i in T.Tiles([block_m], parallel=True):
                row_sum[i] = T.float32(0)

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_square[i, j] = a_shared[i, j] * a_shared[i, j]

            T.reduce(a_square, row_sum, "sum", dim=1, clear=False)

            for i in T.Tiles([block_m], parallel=True):
                row_scale[i] = T.rsqrt(row_sum[i] / T.float32(block_n) + T.float32(eps))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_out[i, j] = a_shared[i, j] * row_scale[i]

            T.copy(a_out, AOut[0:block_m, 0:block_n])

    return main


def rms_norm_tiled_fill_init_kernel(block_m=32, block_n=32, dtype="float32", accum_dtype="float32"):
    eps = 1e-6

    @T.prim_func
    def main(AIn: T.Tensor((block_m, block_n), dtype), AOut: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            a_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            a_square = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            a_out = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            row_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            row_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AIn[0:block_m, 0:block_n], a_shared)
            T.fill(row_sum, T.float32(0))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_square[i, j] = a_shared[i, j] * a_shared[i, j]

            T.reduce(a_square, row_sum, "sum", dim=1, clear=False)

            for i in T.Tiles([block_m], parallel=True):
                row_scale[i] = T.rsqrt(row_sum[i] / T.float32(block_n) + T.float32(eps))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_out[i, j] = a_shared[i, j] * row_scale[i]

            T.copy(a_out, AOut[0:block_m, 0:block_n])

    return main


def rms_norm_tiled_rsram_copy_init_kernel(block_m=32, block_n=32, dtype="float32", accum_dtype="float32"):
    eps = 1e-6

    @T.prim_func
    def main(AIn: T.Tensor((block_m, block_n), dtype), AOut: T.Tensor((block_m, block_n), dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            a_shared = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            a_square = T.alloc_shared((block_m, block_n), accum_dtype, scope="shared.rsram")
            a_out = T.alloc_shared((block_m, block_n), dtype, scope="shared.rsram")
            row_sum = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            row_scale = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")
            row_zero = T.alloc_shared((block_m,), accum_dtype, scope="shared.rsram")

            T.copy(AIn[0:block_m, 0:block_n], a_shared)
            T.fill(row_zero, T.float32(0))
            T.copy(row_zero, row_sum)

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_square[i, j] = a_shared[i, j] * a_shared[i, j]

            T.reduce(a_square, row_sum, "sum", dim=1, clear=False)

            for i in T.Tiles([block_m], parallel=True):
                row_scale[i] = T.rsqrt(row_sum[i] / T.float32(block_n) + T.float32(eps))

            for i, j in T.Tiles([block_m, block_n], parallel=True):
                a_out[i, j] = a_shared[i, j] * row_scale[i]

            T.copy(a_out, AOut[0:block_m, 0:block_n])

    return main


def overwritten_tile_scope_kernel(block_m=32, block_n=32, tile_size=(8, 32), dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((block_m, block_n), dtype),
        B: T.Tensor((block_m, block_n), dtype),
        C: T.Tensor((block_m, block_n), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)
            Tmp_shared = T.alloc_shared((block_m, block_n), dtype)
            C_shared = T.alloc_shared((block_m, block_n), dtype)

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                    Tmp_shared: make_blockwise_zz_layout(Tmp_shared),
                    C_shared: make_blockwise_zz_layout(C_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, (-2, -1)),
                    B_shared: make_tileview(B_shared, tile_size, (-2, -1)),
                    Tmp_shared: make_tileview(Tmp_shared, tile_size, (-2, -1)),
                    C_shared: make_tileview(C_shared, tile_size, (-2, -1)),
                }
            )

            T.copy(A[0:block_m, 0:block_n], A_shared)
            T.copy(B[0:block_m, 0:block_n], B_shared)
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                Tmp_shared[i, j] = A_shared[i, j]
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                Tmp_shared[i, j] = B_shared[i, j]
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                C_shared[i, j] = Tmp_shared[i, j]
            T.copy(C_shared, C[0:block_m, 0:block_n])

    return main


def read_then_overwrite_tile_scope_kernel(block_m=32, block_n=32, tile_size=(8, 32), dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((block_m, block_n), dtype),
        Init: T.Tensor((block_m, block_n), dtype),
        B: T.Tensor((block_m, block_n), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((block_m, block_n), dtype)
            Tmp_shared = T.alloc_shared((block_m, block_n), dtype)
            B_shared = T.alloc_shared((block_m, block_n), dtype)

            T.annotate_layout(
                {
                    A_shared: make_blockwise_zz_layout(A_shared),
                    Tmp_shared: make_blockwise_zz_layout(Tmp_shared),
                    B_shared: make_blockwise_zz_layout(B_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, tile_size, (-2, -1)),
                    Tmp_shared: make_tileview(Tmp_shared, tile_size, (-2, -1)),
                    B_shared: make_tileview(B_shared, tile_size, (-2, -1)),
                }
            )

            T.copy(A[0:block_m, 0:block_n], A_shared)
            T.copy(Init[0:block_m, 0:block_n], Tmp_shared)
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                B_shared[i, j] = Tmp_shared[i, j]
            for i, j in T.Tiles([block_m, block_n], parallel=True):
                Tmp_shared[i, j] = A_shared[i, j]
            T.copy(B_shared, B[0:block_m, 0:block_n])

    return main


def long_row_chain_kernel(num_regions=17, block_m=32, dtype="float32"):
    lines = [
        "@T.prim_func",
        "def main(A: T.Tensor((block_m,), dtype), B: T.Tensor((block_m,), dtype)):",
        "    with T.Kernel(1, threads=128) as (bx,):",
        '        buffers = T.alloc_shared((num_regions + 1, block_m), dtype, scope="shared.rsram")',
        "",
        "        for i in T.Tiles([block_m], parallel=True):",
        "            buffers[0, i] = A[i]",
        "",
    ]
    for region_index in range(num_regions):
        lines.extend(
            [
                "        for i in T.Tiles([block_m], parallel=True):",
                f"            buffers[{region_index + 1}, i] = buffers[{region_index}, i]",
                "",
            ]
        )
    lines.extend(
        [
            "        for i in T.Tiles([block_m], parallel=True):",
            "            B[i] = buffers[num_regions, i]",
        ]
    )
    source = "\n".join(lines)
    filename = f"<generated_long_row_chain_kernel_{num_regions}_{block_m}>"
    linecache.cache[filename] = (len(source), None, [line + "\n" for line in source.splitlines()], filename)
    namespace = {"T": T, "block_m": block_m, "dtype": dtype, "num_regions": num_regions}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["main"]


def attr_wrapped_two_region_lowered_kernel(dtype="float16"):
    @T.prim_func
    def main():
        with T.block("root"):
            T.reads()
            T.writes()
            A_shared = T.alloc_buffer((32, 32), dtype, scope="shared.rsram")
            Tmp_shared = T.alloc_buffer((32, 32), dtype, scope="shared.rsram")
            B_shared = T.alloc_buffer((32, 32), dtype, scope="shared.rsram")

            with T.attr("wrapper_scope", "unit_test", 1):
                for i in T.serial(
                    4,
                    annotations={
                        "tile.domain": [T.int32(32), T.int32(32)],
                        "tile.execution_axis": T.int32(0),
                        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                        "tile.scope_entry": T.int32(1),
                        "tile.tile_size": [T.int32(8), T.int32(32)],
                    },
                ):
                    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
                        for ki in T.serial(8, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}):
                            for kj in T.vectorized(32, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}):
                                Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj]

            with T.attr("wrapper_scope", "unit_test", 1):
                for i in T.serial(
                    4,
                    annotations={
                        "tile.domain": [T.int32(32), T.int32(32)],
                        "tile.execution_axis": T.int32(0),
                        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
                        "tile.scope_entry": T.int32(1),
                        "tile.tile_size": [T.int32(8), T.int32(32)],
                    },
                ):
                    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
                        for ki in T.serial(8, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}):
                            for kj in T.vectorized(32, annotations={"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}):
                                B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj]

    return main


def let_wrapped_two_region_lowered_kernel(dtype="float16"):
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def main():
    A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    x0: T.int32 = 7
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj] + T.Cast("{dtype}", x0)
    x1: T.int32 = 11
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj] + T.Cast("{dtype}", x1)
"""
    return tvm.script.from_source(source)


def mixed_plain_and_let_wrapped_two_region_lowered_kernel(dtype="float16"):
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def main():
    A_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    Tmp_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    B_shared = T.alloc_buffer((32, 32), "{dtype}", scope="shared.rsram")
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj]
    x1: T.int32 = 7
    for i in T.serial(4, annotations={{"tile.domain": [T.int32(32), T.int32(32)], "tile.execution_axis": T.int32(0), "tile.execution_domain_axes": [T.int32(0), T.int32(1)], "tile.scope_entry": T.int32(1), "tile.tile_size": [T.int32(8), T.int32(32)]}}):
        for j in T.serial(1, annotations={{"tile.execution_axis": T.int32(1)}}):
            for ki in T.serial(8, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(0)}}):
                for kj in T.vectorized(32, annotations={{"tile.interior": T.int32(1), "tile.interior_axis": T.int32(1)}}):
                    B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj] + T.Cast("{dtype}", x1)
"""
    return tvm.script.from_source(source)
