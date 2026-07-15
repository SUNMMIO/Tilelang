import json
import os
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
_TEST_OUTPUT_DIR = Path(
    "/home/yesimeng/Tilelang/testing/python/transform/"
    "out_sunmmio_pipeline_strict"
)
_SCHEDULER_LOG_NAMES = (
    "prologue.log",
    "body.log",
    "epilogue.log",
    "body_graph.log",
)


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


class _ScopedWorkingDirectory:
    def __init__(self, path: Path):
        self._path = path
        self._old_path = None

    def __enter__(self):
        self._path.mkdir(parents=True, exist_ok=True)
        self._old_path = Path.cwd()
        os.chdir(self._path)
        return self

    def __exit__(self, exc_type, exc, tb):
        os.chdir(self._old_path)


def _annotation_to_python(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tir.Buffer):
        return {
            "name": value.name,
            "shape": [str(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "scope": value.scope(),
        }
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
        for key, item in value.items():
            if isinstance(key, tir.Buffer):
                key = key.name
            else:
                key = getattr(key, "name", str(key))
            converted[str(key)] = _annotation_to_python(item)
        return converted
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [_annotation_to_python(item) for item in value]
        except Exception:
            return str(value)
    return str(value)


def _extract_pipeline_annotations(stmt):
    result = None

    def visit(node):
        nonlocal result
        if result is not None or node is None:
            return
        if isinstance(node, tir.For):
            annotations = node.annotations
            if annotations and "prologue_orders" in annotations and "body_orders" in annotations:
                result = annotations
                return
            visit(node.body)
        elif isinstance(node, tir.BlockRealize):
            visit(node.block.body)
        elif isinstance(node, tir.Block):
            visit(node.body)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case)
            visit(node.else_case)
        elif isinstance(node, tir.LetStmt):
            visit(node.body)
        elif isinstance(node, tir.AttrStmt):
            visit(node.body)

    visit(stmt)
    return result


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_ir(path: Path, mod: tvm.IRModule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        mod.script(show_meta=True).strip() + "\n", encoding="utf-8"
    )


def _func_attrs(mod: tvm.IRModule):
    func = mod["main"]
    if func.attrs is None:
        return {}
    return {
        str(key): _annotation_to_python(value)
        for key, value in func.attrs.items()
    }


def _artifact_paths(case_name: str) -> dict[str, Path]:
    case_dir = _TEST_OUTPUT_DIR / case_name
    return {
        "case_dir": case_dir,
        "before_pipeline_ir": case_dir / "00_before_pipeline_ir.py",
        "before_func_attrs": case_dir / "00a_before_func_attrs.json",
        "after_planning_ir": case_dir / "01_after_pipeline_planning_ir.py",
        "planning_annotations": case_dir / "01a_pipeline_annotations.json",
        "after_planning_func_attrs": case_dir / "01b_after_planning_func_attrs.json",
        "after_inject_ir": case_dir / "02_after_inject_pipeline_ir.py",
        "after_inject_func_attrs": case_dir / "02a_after_inject_func_attrs.json",
        "manifest": case_dir / "README.json",
    }


def _buffer_names(annotations, key):
    if key not in annotations:
        return []
    return sorted(buffer.name for buffer in annotations[key])


def _validate_orders(annotations, instruction_count):
    for key in ("prologue_orders", "body_orders", "epilogue_orders"):
        if key not in annotations:
            continue
        seen = set()
        for value in annotations[key]:
            text = str(value)
            parts = text.split("-")
            assert len(parts) == 2, (key, text)
            iteration, statement_id = (int(part) for part in parts)
            assert iteration >= 0, (key, text)
            assert 0 <= statement_id < instruction_count, (key, text)
            assert text not in seen, (key, text)
            seen.add(text)


def _pipeline_instruction_count(stmt):
    annotations = _extract_pipeline_annotations(stmt)
    assert annotations is not None

    result = None

    def visit(node):
        nonlocal result
        if result is not None or node is None:
            return
        if isinstance(node, tir.For):
            if node.annotations.same_as(annotations):
                body = node.body
                while isinstance(body, (tir.BlockRealize, tir.Block, tir.LetStmt, tir.AttrStmt)):
                    if isinstance(body, tir.BlockRealize):
                        body = body.block.body
                    else:
                        body = body.body
                assert isinstance(body, tir.SeqStmt), type(body)
                result = len(body.seq)
                return
            visit(node.body)
        elif isinstance(node, tir.BlockRealize):
            visit(node.block.body)
        elif isinstance(node, tir.Block):
            visit(node.body)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case)
            visit(node.else_case)
        elif isinstance(node, tir.LetStmt):
            visit(node.body)
        elif isinstance(node, tir.AttrStmt):
            visit(node.body)

    visit(stmt)
    assert result is not None
    return result


def _build_and_record(case_name, kernel_factory, requested_num_stages):
    artifacts = _artifact_paths(case_name)
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = tvm.IRModule.from_expr(
            kernel_factory().with_attr("global_symbol", "main")
        )
        mod = lower_and_legalize_sunmmio_pipeline_test(mod, target)
        mod = tl.transform.IfStmtBinding()(mod)
        _write_ir(artifacts["before_pipeline_ir"], mod)
        _write_json(artifacts["before_func_attrs"], _func_attrs(mod))

        for log_name in _SCHEDULER_LOG_NAMES:
            log_path = artifacts["case_dir"] / log_name
            if log_path.exists():
                log_path.unlink()
        with _ScopedWorkingDirectory(artifacts["case_dir"]):
            planned = tl.transform.SunmmioPipelinePlanning(debug=True)(mod)

        annotations = _extract_pipeline_annotations(planned["main"].body)
        assert annotations is not None, case_name
        annotation_payload = {
            str(key): _annotation_to_python(value)
            for key, value in annotations.items()
        }
        _write_ir(artifacts["after_planning_ir"], planned)
        _write_json(artifacts["planning_annotations"], annotation_payload)
        _write_json(
            artifacts["after_planning_func_attrs"], _func_attrs(planned)
        )

        injected = tl.transform.InjectSunmmioPipeline()(planned)
        _write_ir(artifacts["after_inject_ir"], injected)
        _write_json(artifacts["after_inject_func_attrs"], _func_attrs(injected))

    scheduler_logs = {
        name: str(artifacts["case_dir"] / name)
        for name in _SCHEDULER_LOG_NAMES
        if (artifacts["case_dir"] / name).exists()
    }
    manifest = {
        "case_name": case_name,
        "pipeline": "greedy",
        "requested_num_stages": requested_num_stages,
        "selected_iterations": annotation_payload.get("iterations"),
        "schedule_sizes": {
            key: len(annotations[key])
            for key in ("prologue_orders", "body_orders", "epilogue_orders")
            if key in annotations
        },
        "versioned_buffers": _buffer_names(annotations, "versioned_buffers"),
        "artifacts": {
            key: str(path)
            for key, path in artifacts.items()
            if key not in ("case_dir", "manifest")
        },
        "scheduler_logs": scheduler_logs,
    }
    _write_json(artifacts["manifest"], manifest)
    return planned, injected, artifacts


STRICT_CASES = [
    (
        "matmul_num_stages_4",
        lambda: mesh_matmul_new(
            1024, 1024, 1024, 128, 128, 32, num_stages=4
        ),
        4,
        {
            "A_rsram_stage": [4, 128, 32],
            "A_shared_ping": [2, 128, 32],
            "A_shared_pong": [2, 128, 32],
            "B_shared_ping": [2, 32, 128],
            "B_shared_pong": [2, 32, 128],
        },
    ),
    (
        "flashattn2",
        lambda: mesh_flashattn_new(num_stages=2),
        2,
        {
            "K_shared_ping": [1, 64, 128],
            "K_shared_pong": [1, 64, 128],
            "acc_s_cast_ping": [1, 64, 64],
            "acc_s_cast_pong": [1, 64, 64],
            "V_shared_ping": [1, 64, 128],
            "V_shared_pong": [1, 64, 128],
        },
    ),
    (
        "flashdecoding2",
        lambda: mesh_flashdecoding_new(num_stages=2),
        2,
        {
            "K_shared_ping": [1, 128, 128],
            "K_shared_pong": [1, 128, 128],
            "V_shared_ping": [1, 128, 128],
            "V_shared_pong": [1, 128, 128],
        },
    ),
    (
        "flashmladecode2",
        lambda: mesh_flashmladecode_new(num_stages=2),
        2,
        {
            "KV_shared_ping": [1, 64, 512],
            "KV_shared_pong": [1, 64, 512],
            "KV_shared2_ping": [1, 64, 512],
            "KV_shared2_pong": [1, 64, 512],
            "K_pe_shared_ping": [1, 64, 64],
            "K_pe_shared_pong": [1, 64, 64],
            "S_shared_ping": [1, 64, 64],
            "S_shared_pong": [1, 64, 64],
        },
    ),
]


@pytest.mark.parametrize(
    "case_name,kernel_factory,num_stages,expected_shapes",
    STRICT_CASES,
    ids=[case[0] for case in STRICT_CASES],
)
def test_tilelang_transform_sunmmio_pipeline_strict(
    case_name, kernel_factory, num_stages, expected_shapes
):
    planned, injected, artifacts = _build_and_record(
        case_name, kernel_factory, num_stages
    )

    annotations = _extract_pipeline_annotations(planned["main"].body)
    assert int(annotations["iterations"]) == num_stages
    assert _buffer_names(annotations, "versioned_buffers")
    instruction_count = _pipeline_instruction_count(planned["main"].body)
    _validate_orders(annotations, instruction_count)

    layout_map = injected["main"].attrs["layout_map"]
    actual_shapes = {
        buffer.name: [int(dim) for dim in buffer.shape]
        for buffer, _ in layout_map.items()
    }
    logical_shapes = {
        buffer.name: [int(dim) for dim in _get_logical_shape(layout)]
        for buffer, layout in layout_map.items()
    }
    for name, expected_shape in expected_shapes.items():
        assert actual_shapes[name] == expected_shape
        assert logical_shapes[name] == expected_shape

    ping_pong = injected["main"].attrs["tl.sunmmio_alloc_ping_pong"]
    assert {str(value) for value in ping_pong.values()} == {"pong"}
    assert all((artifacts["case_dir"] / name).exists() for name in _SCHEDULER_LOG_NAMES)
    assert artifacts["manifest"].exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
