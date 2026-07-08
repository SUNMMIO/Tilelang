import os
import json
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

_get_logical_shape = tvm.ffi.get_global_func("tl.CuteLayout_logical_shape")

_TEST_OUTPUT_DIR = Path("/home/yesimeng/Tilelang/testing/python/transform/out_sunmmio_pipeline_strict_ilp")


def lower_and_legalize_sunmmio_pipeline_test(mod, target):
    mod = tir.transform.BindTarget(target)(mod)
    if should_force_let_inline():
        mod = tl.transform.LetInline()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.InferSramScope()(mod)
    mod = tl.transform.LegalizeSunmmioDataPath()(mod)
    mod = tl.transform.SunmmioLayoutInference()(mod)
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
                "prologue_orders" in ann
                or "body_orders" in ann
                or "epilogue_orders" in ann
                or "runtime_multiversion_buffers" in ann
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
        elif isinstance(node, tir.LetStmt):
            visit(node.body)
        elif isinstance(node, tir.AttrStmt):
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
        text = text.replace(f"T.Mul(0, {step})", f"__iter__ * {step}")
    return text


def _runtime_stmt_matches_order(runtime_script, planned_stmt, iter_kind, iter_offset):
    planned_script = _apply_iter_offset(
        _normalize_planned_stmt(_stmt_script(planned_stmt)),
        iter_offset,
    )
    runtime_script = _normalize_runtime_stmt(runtime_script)
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
                body = stmt.body
                if isinstance(body, tir.SeqStmt) and len(body.seq) == 1 and isinstance(body.seq[0], tir.IfThenElse):
                    return node, idx, stmt, body.seq[0]
                if isinstance(body, tir.IfThenElse):
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
    steady_parent, steady_idx, steady_loop, steady_if = injected_found

    prologue_scripts = [_stmt_script(steady_parent.seq[i]) for i in range(steady_idx)]
    _find_ordered_matches(prologue_scripts, annotations["prologue_orders"], planned_body, "prologue")

    then_scripts = _steady_branch_scripts(steady_if.then_case)
    else_scripts = _steady_branch_scripts(steady_if.else_case)
    body_orders = [str(v) for v in annotations["body_orders"]]
    _find_ordered_matches(then_scripts, body_orders, planned_body, "steady")
    _find_ordered_matches(else_scripts, body_orders, planned_body, "steady")

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
        with tl.transform.PassContext(
            config={tl.PassConfigKey.TL_SUNMMIO_ILP_STAGE_SHRINK: True}
        ):
            with _ScopedEnv(
                {
                    "TL_SUNMMIO_ILP_FASTER": "20",
                    "TL_SUNMMIO_ILP_PROBLEM_JSON": str(artifacts["problem_json"]),
                    "TL_SUNMMIO_ILP_SOLUTION_JSON": str(artifacts["solution_json"]),
                }
            ):
                planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
        _write_ir(artifacts["after_planning_ir"], planned)
        _write_pipeline_annotations(artifacts["planning_annotations"], planned)
        injected = tl.transform.InjectSunmmioPipelineILP()(planned)
        _write_ir(artifacts["after_inject_ir"], injected)

    stage_attempts_payload = {}
    annotations = _extract_pipeline_annotations(planned["main"].body)
    if annotations is not None:
        anno_payload = {
            str(key): _annotation_to_python(value)
            for key, value in annotations.items()
        }
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
        lambda: mesh_flashattn_new(num_stages=2),
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
        lambda: mesh_flashdecoding_new(num_stages=2),
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
        lambda: mesh_flashmladecode_new(num_stages=2),
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
    setattr(_kernel_factory, "_strict_case_name", _case_name)
    setattr(_kernel_factory, "_requested_num_stages", 3)


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
    assert _annotation_buffer_names(annotations, "runtime_multiversion_buffers") == [], case_name
    assert len(_annotation_buffer_names(annotations, "runtime_banked_buffers")) > 0, case_name

    _check_order_mapping(planned, injected)

    func = injected["main"]
    _assert_multiversioned_func_layouts(func, expected_shapes)
    _assert_layout_names_absent(func, forbidden_layout_names)

    if "tl.sunmmio_alloc_ping_pong" in func.attrs:
        ping_pong = _annotation_to_python(func.attrs["tl.sunmmio_alloc_ping_pong"])
        assert set(ping_pong.values()) == {"pong"}, case_name
