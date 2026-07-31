import os
import json
import re
from pathlib import Path

import pytest
import tilelang as tl
from tilelang import tvm
from tilelang.engine.phase import should_force_let_inline
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from testing.python.transform.sunmmio_mesh_kernel_new_syntax_reference import (
    mesh_flashattn_new,
    mesh_flashdecoding_new,
    mesh_flashmladecode_new,
    mesh_matmul_new,
)
from tvm import tir
from tvm.tir.stmt_functor import ir_transform

_get_logical_shape = tvm.ffi.get_global_func("tl.CuteLayout_logical_shape")

_TEST_OUTPUT_DIR = Path("/home/yesimeng/Tilelang/testing/python/transform/out_sunmmio_pipeline_strict_ilp")


def lower_and_legalize_sunmmio_pipeline_test(mod, target):
    mod = tir.transform.BindTarget(target)(mod)
    mod = tl.transform.ResolveSunmmioMeshSymbols()(mod)
    if should_force_let_inline():
        mod = tl.transform.LetInline()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.InferSramScope()(mod)
    mod = tl.transform.LegalizeSunmmioDataPath()(mod)
    mod = tl.transform.SunmmioLayoutInference()(mod)
    mod = tl.transform.LegalizeSunmmioGemm()(mod)
    mod = tl.transform.LowerTileOp()(mod)
    mod = tl.transform.LegalizeTilesLoop()(mod)
    mod = tl.transform.TilesLoop()(mod)
    mod = tl.transform.LegalizeVectorizedLoop()(mod)
    mod = tl.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tl.transform.LowerAccessPtr()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.HoistNonRestrictParams()(mod)
    mod = tl.transform.HoistBlockAnnotationsToFuncAttrs()(mod)
    return mod


class _ScopedEnv:
    def __init__(self, updates):
        self._updates = updates
        self._old = {}

    def __enter__(self):
        for key, value in self._updates.items():
            self._old[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, old_value in self._old.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _annotation_to_python(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        try:
            return int(value.value)
        except Exception:
            try:
                return float(value.value)
            except Exception:
                return str(value)
    if hasattr(value, "items"):
        converted = {}
        for k, v in value.items():
            key = getattr(k, "name", None)
            if key is None:
                key = str(k)
            converted[key] = _annotation_to_python(v)
        return converted
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [_annotation_to_python(v) for v in value]
        except Exception:
            return str(value)
    return str(value)


def _extract_pipeline_annotations(stmt):
    result = None

    def visit(node):
        nonlocal result
        if result is not None:
            return
        if isinstance(node, tir.For):
            ann = node.annotations
            if ann and (
                "prologue_orders" in ann or "body_orders" in ann or "epilogue_orders" in ann or "runtime_multiversion_buffers" in ann
            ):
                result = ann
                return
        if isinstance(node, tir.For):
            visit(node.body)
        elif isinstance(node, tir.BlockRealize):
            visit(node.block.body)
        elif isinstance(node, tir.Block):
            visit(node.body)
        elif isinstance(node, tir.SeqStmt):
            for s in node.seq:
                visit(s)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case)
            if node.else_case is not None:
                visit(node.else_case)
        elif isinstance(node, (tir.LetStmt, tir.AttrStmt)):
            visit(node.body)

    visit(stmt)
    return result


def _annotation_buffer_names(annotations, key):
    if key not in annotations:
        return []
    return sorted(buffer.name for buffer in annotations[key])


def _write_ir(path: Path, mod: tvm.IRModule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mod.script(show_meta=True).strip() + "\n")


def _write_pipeline_annotations(path: Path, mod: tvm.IRModule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    func = mod["main"]
    annotations = _extract_pipeline_annotations(func.body)
    payload = {}
    if annotations is not None:
        for key, value in annotations.items():
            payload[str(key)] = _annotation_to_python(value)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_paths(case_name: str) -> dict[str, Path]:
    case_dir = _TEST_OUTPUT_DIR / case_name
    return {
        "case_dir": case_dir,
        "before_pipeline_ir": case_dir / "00_before_pipeline_ir.py",
        "after_planning_ir": case_dir / "01_after_pipeline_planning_ilp_ir.py",
        "planning_annotations": case_dir / "01a_pipeline_annotations.json",
        "after_inject_ir": case_dir / "02_after_inject_pipeline_ilp_ir.py",
        "problem_json": case_dir / "ilp_problem.json",
        "solution_json": case_dir / "ilp_solution.json",
        "stage_attempts": case_dir / "stage_attempts.json",
    }


def _shape_to_int_list(shape):
    return [int(dim) for dim in shape]


def _assert_multiversioned_func_layouts(func, expected_shapes):
    assert "layout_map" in func.attrs
    layout_map = func.attrs["layout_map"]

    layout_shapes = {}
    buffer_shapes = {}
    for buffer, layout in layout_map.items():
        layout_shapes[buffer.name] = _shape_to_int_list(_get_logical_shape(layout))
        buffer_shapes[buffer.name] = _shape_to_int_list(buffer.shape)

    for buffer_name, expected_shape in expected_shapes.items():
        expected_shape = list(expected_shape)
        assert buffer_name in layout_shapes, layout_shapes
        assert layout_shapes[buffer_name] == expected_shape, layout_shapes
        assert buffer_shapes[buffer_name] == expected_shape, buffer_shapes


def _assert_layout_names_absent(func, forbidden_names):
    assert "layout_map" in func.attrs
    layout_names = {buffer.name for buffer, _ in func.attrs["layout_map"].items()}
    for name in forbidden_names:
        assert name not in layout_names, layout_names


def _stmt_script(stmt):
    script = stmt.script() if hasattr(stmt, "script") else str(stmt)
    return " ".join(script.split())


def _int_value(expr):
    return int(getattr(expr, "value", expr))


def _get_pipeline_for(node):
    if isinstance(node, tir.For):
        ann = node.annotations
        if ann and "body_orders" in ann and "prologue_orders" in ann and "epilogue_orders" in ann:
            return node
        return _get_pipeline_for(node.body)
    if isinstance(node, tir.BlockRealize):
        return _get_pipeline_for(node.block.body)
    if isinstance(node, tir.Block):
        return _get_pipeline_for(node.body)
    if isinstance(node, tir.SeqStmt):
        for stmt in node.seq:
            found = _get_pipeline_for(stmt)
            if found is not None:
                return found
        return None
    if isinstance(node, tir.IfThenElse):
        found = _get_pipeline_for(node.then_case)
        if found is not None:
            return found
        if node.else_case is not None:
            return _get_pipeline_for(node.else_case)
        return None
    if isinstance(node, tir.LetStmt):
        return _get_pipeline_for(node.body)
    if isinstance(node, tir.AttrStmt):
        return _get_pipeline_for(node.body)
    return None


def _unwrap_seq(stmt):
    if isinstance(stmt, tir.BlockRealize):
        return _unwrap_seq(stmt.block.body)
    if isinstance(stmt, tir.Block):
        return _unwrap_seq(stmt.body)
    if isinstance(stmt, tir.SeqStmt):
        return stmt
    return None


def _pipeline_body_seq(for_node):
    seq = _unwrap_seq(for_node.body)
    assert seq is not None, f"expected pipeline loop body to be SeqStmt, got {type(for_node.body)}"
    return list(seq.seq)


def _normalize_runtime_stmt(script):
    text = " ".join(script.split())
    for token in ("_ping", "_pong"):
        text = text.replace(token, "")
    # KV_shared2 uses a leading runtime lifetime-version axis in addition to
    # its physical ping/pong bank.  Remove only that axis when comparing the
    # injected statement with the original two-dimensional command.
    text = re.sub(
        r"KV_shared2\[[^,\]]+, ([^,\]]+), ([^\]]+)\]",
        r"KV_shared2[\1, \2]",
        text,
    )
    text = re.sub(
        r"(T\.region\(KV_shared2\[[^\]]+\], [12]), 1, ",
        r"\1, ",
        text,
    )
    return _semantic_tail(text)


def _normalize_planned_stmt(script):
    text = " ".join(script.split())
    return _semantic_tail(text)


def _semantic_tail(text):
    anchors = [
        "T.dma_copy(",
        "T.mma_sunmmio(",
        "T.sunmmio_layout_transform(",
        "T.vector_core_in_tile_reduce(",
        "= T.if_then_else(",
        "= T.max(",
        "= T.Cast(",
        "= T.exp2(",
        "= T.log2(",
        "= T.infinity(",
        "= T.bfloat16(",
        "= T.float32(",
    ]
    positions = [text.find(anchor) for anchor in anchors if text.find(anchor) != -1]
    if not positions:
        return text
    return text[min(positions) :]


def _apply_iter_offset(text, iter_offset):
    iter_expr = "__iter__" if iter_offset == 0 else f"(__iter__ + {iter_offset})"
    for step in (32, 64, 128):
        text = text.replace(f"k * {step}", f"{iter_expr} * {step}")
        text = text.replace(f"(k + 1) * {step}", f"(__iter__ + 1) * {step}")
        text = text.replace(f"(k + 2) * {step}", f"(__iter__ + 2) * {step}")
        text = text.replace(f"T.Mul({iter_offset}, {step})", f"{iter_expr} * {step}")
        text = text.replace(f"T.Mul(0, {step})", f"__iter__ * {step}")
    return text


def _runtime_stmt_matches_order(runtime_script, planned_stmt, iter_kind, iter_offset):
    planned_script = _apply_iter_offset(
        _normalize_planned_stmt(_stmt_script(planned_stmt)),
        iter_offset,
    )
    runtime_script = _normalize_runtime_stmt(runtime_script)
    if iter_kind in ("steady-even", "steady-odd"):
        base_shift = 0 if iter_kind == "steady-even" else 1
        # Normalize the statically unrolled ping/pong super-iteration back to
        # the logical base used by the planned statement.
        for total_offset in range(8, 0, -1):
            logical_offset = total_offset - base_shift
            if logical_offset < 0:
                continue
            replacement = "__iter__" if logical_offset == 0 else f"(__iter__ + {logical_offset})"
            runtime_script = runtime_script.replace("2 * k" + " + 1" * total_offset, replacement)
        runtime_script = runtime_script.replace("2 * k", "__iter__")
        runtime_script = re.sub(r"\(\((__iter__ \+ [0-9]+)\)\)", r"(\1)", runtime_script)
    runtime_script = _apply_iter_offset(runtime_script, iter_offset)
    return planned_script in runtime_script


def _find_ordered_matches(runtime_scripts, order_strings, planned_body, iter_kind):
    matched_indices = []
    cursor = 0
    for order_str in order_strings:
        iter_offset = int(str(order_str).split("-")[0])
        stmt_id = int(str(order_str).split("-")[1])
        planned_stmt = planned_body[stmt_id]
        found = False
        while cursor < len(runtime_scripts):
            if _runtime_stmt_matches_order(runtime_scripts[cursor], planned_stmt, iter_kind, iter_offset):
                matched_indices.append(cursor)
                cursor += 1
                found = True
                break
            cursor += 1
        assert found, (
            order_str,
            iter_kind,
            _stmt_script(planned_stmt),
            runtime_scripts,
        )
    return matched_indices


def _collect_top_level_scripts(node):
    seq = _unwrap_seq(node)
    if seq is None:
        return [_stmt_script(node)]
    return [_stmt_script(stmt) for stmt in seq.seq]


def _find_steady_loop_parent(node):
    if isinstance(node, tir.SeqStmt):
        for idx, stmt in enumerate(node.seq):
            if isinstance(stmt, tir.For) and stmt.loop_var.name == "k":
                body = _unwrap_seq(stmt.body)
                if body is not None:
                    return node, idx, stmt, body
            found = _find_steady_loop_parent(stmt)
            if found is not None:
                return found
        return None
    if isinstance(node, tir.BlockRealize):
        return _find_steady_loop_parent(node.block.body)
    if isinstance(node, tir.Block):
        return _find_steady_loop_parent(node.body)
    if isinstance(node, tir.For):
        return _find_steady_loop_parent(node.body)
    if isinstance(node, tir.IfThenElse):
        found = _find_steady_loop_parent(node.then_case)
        if found is not None:
            return found
        if node.else_case is not None:
            return _find_steady_loop_parent(node.else_case)
        return None
    if isinstance(node, tir.LetStmt):
        return _find_steady_loop_parent(node.body)
    if isinstance(node, tir.AttrStmt):
        return _find_steady_loop_parent(node.body)
    return None


def _steady_branch_scripts(branch):
    seq = _unwrap_seq(branch)
    assert seq is not None
    return [_stmt_script(stmt) for stmt in seq.seq]


def _check_order_mapping(planned, injected):
    annotations = _extract_pipeline_annotations(planned["main"].body)
    assert annotations is not None

    planned_pipeline = _get_pipeline_for(planned["main"].body)
    assert planned_pipeline is not None
    planned_body = _pipeline_body_seq(planned_pipeline)

    injected_found = _find_steady_loop_parent(injected["main"].body)
    assert injected_found is not None
    steady_parent, steady_idx, steady_loop, steady_body = injected_found

    prologue_scripts = [_stmt_script(steady_parent.seq[i]) for i in range(steady_idx)]
    _find_ordered_matches(prologue_scripts, annotations["prologue_orders"], planned_body, "prologue")

    body_orders = [str(v) for v in annotations["body_orders"]]
    steady_scripts = [_stmt_script(stmt) for stmt in steady_body.seq]
    assert len(steady_scripts) == 2 * len(body_orders)
    _find_ordered_matches(steady_scripts[: len(body_orders)], body_orders, planned_body, "steady-even")
    _find_ordered_matches(steady_scripts[len(body_orders) :], body_orders, planned_body, "steady-odd")

    epilogue_scripts = [_stmt_script(stmt) for stmt in steady_parent.seq[steady_idx + 1 :]]
    _find_ordered_matches(epilogue_scripts, annotations["epilogue_orders"], planned_body, "steady")


def _build_pipeline_modules(kernel_factory):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    case_name = getattr(kernel_factory, "_strict_case_name", "unknown")
    artifacts = _artifact_paths(case_name)
    requested_num_stages = getattr(kernel_factory, "_requested_num_stages", None)
    with tvm.target.Target(target):
        mod = tvm.IRModule.from_expr(kernel_factory().with_attr("global_symbol", "main"))
        mod = lower_and_legalize_sunmmio_pipeline_test(mod, target)
        mod = tl.transform.IfStmtBinding()(mod)
        _write_ir(artifacts["before_pipeline_ir"], mod)
        with (
            tl.transform.PassContext(config={tl.PassConfigKey.TL_SUNMMIO_ILP_STAGE_SHRINK: False}),
            _ScopedEnv(
                {
                    "TL_SUNMMIO_FASTER": "50",
                    "TL_SUNMMIO_ILP_PROBLEM_JSON": str(artifacts["problem_json"]),
                    "TL_SUNMMIO_ILP_SOLUTION_JSON": str(artifacts["solution_json"]),
                }
            ),
        ):
            planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
        _write_ir(artifacts["after_planning_ir"], planned)
        _write_pipeline_annotations(artifacts["planning_annotations"], planned)
        injected = tl.transform.InjectSunmmioPipelineILP()(planned)
        _write_ir(artifacts["after_inject_ir"], injected)

    stage_attempts_payload = {}
    annotations = _extract_pipeline_annotations(planned["main"].body)
    if annotations is not None:
        anno_payload = {str(key): _annotation_to_python(value) for key, value in annotations.items()}
        stage_attempts_payload.update(
            {
                "stage_count": anno_payload.get("stage_count"),
                "iterations": anno_payload.get("iterations"),
                "steady_state_max_iter_offset": anno_payload.get("steady_state_max_iter_offset"),
                "runtime_banked_buffers": _annotation_buffer_names(annotations, "runtime_banked_buffers"),
                "versioned_buffers": _annotation_buffer_names(annotations, "versioned_buffers"),
                "selected_stage_count": anno_payload.get("stage_count"),
            }
        )
    solution_path = artifacts["solution_json"]
    if solution_path.exists():
        try:
            solution_payload = json.loads(solution_path.read_text(encoding="utf-8"))
            stage_attempts_payload.update(
                {
                    "ii": solution_payload.get("ii"),
                    "makespan": solution_payload.get("makespan"),
                }
            )
        except Exception:
            pass
    stage_attempts_payload.update(
        {
            "requested_num_stages": requested_num_stages,
            "problem_json": str(artifacts["problem_json"]),
            "solution_json": str(artifacts["solution_json"]),
            "before_pipeline_ir": str(artifacts["before_pipeline_ir"]),
            "after_planning_ir": str(artifacts["after_planning_ir"]),
            "planning_annotations": str(artifacts["planning_annotations"]),
            "after_inject_ir": str(artifacts["after_inject_ir"]),
        }
    )
    artifacts["stage_attempts"].write_text(
        json.dumps(stage_attempts_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return planned, injected


STRICT_CASES = [
    # (
    #     "ilp",
    #     lambda: ilp(),{
    #         "A_rsram_stage": [128, 32],
    #         "A_shared_ping": [128, 32],
    #         "A_shared_pong": [128, 32],
    #         "B_shared_ping": [32, 128],
    #         "B_shared_pong": [32, 128],
    #     },
    #     ["A_shared", "B_shared"],
    # ),
    (
        "matmul2",
        lambda: mesh_matmul_new(1024, 1024, 1024, 128, 128, 32, num_stages=3),
        {
            "A_rsram_stage": [128, 32],
            "A_shared_ping": [128, 32],
            "A_shared_pong": [128, 32],
            "B_shared_ping": [32, 128],
            "B_shared_pong": [32, 128],
        },
        ["A_shared", "B_shared"],
    ),
    (
        "flashattn2",
        lambda: mesh_flashattn_new(num_stages=4),
        {
            "K_shared_ping": [64, 128],
            "K_shared_pong": [64, 128],
            "acc_s_cast_ping": [64, 64],
            "acc_s_cast_pong": [64, 64],
            "V_shared_ping": [64, 128],
            "V_shared_pong": [64, 128],
        },
        ["K_shared", "V_shared", "acc_s_cast"],
    ),
    (
        "flashdecoding2",
        lambda: mesh_flashdecoding_new(num_stages=3),
        {
            "K_shared_ping": [128, 128],
            "K_shared_pong": [128, 128],
            "acc_s_cast_ping": [64, 128],
            "acc_s_cast_pong": [64, 128],
            "V_shared_ping": [128, 128],
            "V_shared_pong": [128, 128],
        },
        ["K_shared", "V_shared", "acc_s_cast"],
    ),
    (
        "flashmladecode2",
        lambda: mesh_flashmladecode_new(num_stages=5),
        {
            "KV_shared_ping": [64, 512],
            "KV_shared_pong": [64, 512],
            "KV_shared2_ping": [64, 512],
            "KV_shared2_pong": [64, 512],
            "K_pe_shared_ping": [64, 64],
            "K_pe_shared_pong": [64, 64],
            "S_shared_ping": [64, 64],
            "S_shared_pong": [64, 64],
        },
        ["KV_shared", "KV_shared2", "K_pe_shared", "S_shared"],
    ),
]

for _case_name, _kernel_factory, _, _ in STRICT_CASES:
    _kernel_factory._strict_case_name = _case_name
    _kernel_factory._requested_num_stages = 3


@pytest.mark.parametrize(
    "case_name,kernel_factory,expected_shapes,forbidden_layout_names",
    STRICT_CASES,
    ids=[case_name for case_name, _, _, _ in STRICT_CASES],
)
def test_tilelang_transform_sunmmio_pipeline_strict_ilptest(
    case_name,
    kernel_factory,
    expected_shapes,
    forbidden_layout_names,
):
    planned, injected = _build_pipeline_modules(kernel_factory)

    annotations = _extract_pipeline_annotations(planned["main"].body)
    assert annotations is not None, case_name
    expected_runtime_multiversion = ["scores_max_prev"] if case_name == "flashmladecode2" else []
    assert _annotation_buffer_names(annotations, "runtime_multiversion_buffers") == expected_runtime_multiversion, case_name
    assert len(_annotation_buffer_names(annotations, "runtime_banked_buffers")) > 0, case_name

    _check_order_mapping(planned, injected)

    func = injected["main"]
    _assert_multiversioned_func_layouts(func, expected_shapes)
    _assert_layout_names_absent(func, forbidden_layout_names)

    if "tl.sunmmio_alloc_ping_pong" in func.attrs:
        ping_pong = _annotation_to_python(func.attrs["tl.sunmmio_alloc_ping_pong"])
        assert set(ping_pong.values()) == {"pong"}, case_name


def test_per_op_phase_offsets_select_matching_region_banks():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        func = mesh_matmul_new(
            512,
            512,
            256,
            block_M=128,
            block_N=128,
            block_K=32,
            num_stages=3,
        ).with_attr("global_symbol", "main")
        mod = tvm.IRModule.from_expr(func)
        mod = lower_and_legalize_sunmmio_pipeline_test(mod, target)
        mod = tl.transform.IfStmtBinding()(mod)
        with _ScopedEnv({"TL_SUNMMIO_FASTER": "20"}):
            planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)

        annotations = _extract_pipeline_annotations(planned["main"].body)
        assert annotations is not None
        writer_offsets = annotations["runtime_bank_writer_phases"]
        reader_offsets = annotations["runtime_bank_reader_phases"]
        flip_modes = annotations["runtime_bank_flip_modes"]
        a_buffer = next(buffer for buffer in writer_offsets if buffer.name == "A_shared")
        writer_phases = {int(op): int(offset) for op, offset in writer_offsets[a_buffer].items()}
        reader_phases = {int(op): int(offset) for op, offset in reader_offsets[a_buffer].items()}
        assert writer_phases.keys() == {1, 4}
        assert reader_phases.keys() == {3, 5}
        assert writer_phases[1] == reader_phases[3]
        assert writer_phases[4] == reader_phases[5]
        assert writer_phases[1] != writer_phases[4]

        injected = tl.transform.InjectSunmmioPipelineILP()(planned)
        script = injected.script(show_meta=True)
        assert "if k % 2 == 0:" not in script
        injected_found = _find_steady_loop_parent(injected["main"].body)
        assert injected_found is not None
        _, _, _, steady_body = injected_found
        body_order_count = len(annotations["body_orders"])
        assert len(steady_body.seq) == 2 * body_order_count
        even_branch = "\n".join(_stmt_script(stmt) for stmt in steady_body.seq[:body_order_count])
        odd_branch = "\n".join(_stmt_script(stmt) for stmt in steady_body.seq[body_order_count:])

        flip = bool(int(flip_modes[a_buffer]))
        # op4 uses logical iteration k + 1. In no-flip mode its phase is fixed;
        # in flip mode the logical iteration parity is XORed into that phase.
        even_bank_index = writer_phases[4] ^ (1 if flip else 0)
        odd_bank_index = writer_phases[4]
        even_bank = "ping" if even_bank_index == 0 else "pong"
        odd_bank = "ping" if odd_bank_index == 0 else "pong"
        assert f"T.region(A_shared_{even_bank}[0, 0], 2, 128, 32), 1024" in even_branch
        assert f"T.region(A_shared_{even_bank}[0, 0], 1, 128, 32)" in even_branch
        assert f"T.region(A_shared_{odd_bank}[0, 0], 2, 128, 32), 1024" in odd_branch
        assert f"T.region(A_shared_{odd_bank}[0, 0], 1, 128, 32)" in odd_branch

        # Global A/B regions are not banked, but their loop indices must still
        # be rewritten from the removed pipeline loop to the steady-state loop.
        tvm.tir.transform.RemoveNoOp()(injected)


def test_ilp_inject_failure_restores_unversioned_serial_loop():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        func = mesh_matmul_new(
            512,
            512,
            256,
            block_M=128,
            block_N=128,
            block_K=32,
            num_stages=3,
        ).with_attr("global_symbol", "main")
        mod = tvm.IRModule.from_expr(func)
        mod = lower_and_legalize_sunmmio_pipeline_test(mod, target)
        mod = tl.transform.IfStmtBinding()(mod)
        with _ScopedEnv({"TL_SUNMMIO_FASTER": "20"}):
            planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)

        corrupted = False

        def corrupt_pipeline_loop(node):
            nonlocal corrupted
            if corrupted or not isinstance(node, tir.For) or "body_orders" not in node.annotations:
                return None
            annotations = dict(node.annotations)
            annotations["body_orders"] = ["1-999"]
            corrupted = True
            return tir.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                annotations,
            )

        planned_func = planned["main"]
        corrupted_body = ir_transform(planned_func.body, None, corrupt_pipeline_loop, ["tir.For"])
        corrupted_func = tir.PrimFunc(
            planned_func.params,
            corrupted_body,
            planned_func.ret_type,
            planned_func.buffer_map,
            planned_func.attrs,
        )
        injected = tl.transform.InjectSunmmioPipelineILP()(tvm.IRModule.from_expr(corrupted_func))

        script = injected.script(show_meta=True)
        assert '"tl.sunmmio.pipeline.applied": T.bool(False)' in script
        assert '"tl.sunmmio.pipeline.fallback_stage": "inject_exception"' in script
        assert '"tl.sunmmio.pipeline.fallback_reason": "candidate_rewrite_failed"' in script
        assert '"body_orders"' not in script
        assert "A_shared_ping" not in script
        assert "A_shared_pong" not in script
