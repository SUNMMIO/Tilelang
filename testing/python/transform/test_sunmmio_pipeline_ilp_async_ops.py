import json
import os
from contextlib import contextmanager

import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir


def _region(buffer, indices, access_mask, extents):
    return tir.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        [tir.BufferLoad(buffer, indices), access_mask, *extents],
    )


def _async_call(op_name, *args):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get(op_name), list(args)))


def _make_layout_transform_pipeline():
    A = tir.decl_buffer((8, 16), "float32", name="A")
    B = tir.decl_buffer((8, 16), "float32", name="B")
    src = tir.decl_buffer((16,), "float32", name="src", scope="shared.rsram")
    dst = tir.decl_buffer((16,), "float32", name="dst", scope="shared.rsram")
    k = tir.Var("k", "int32")

    body = tir.SeqStmt(
        [
            _async_call(
                "tl.dma_copy",
                _region(A, [k, 0], 1, [1, 16]),
                _region(src, [0], 2, [16]),
                0,
            ),
            _async_call(
                "tl.sunmmio_layout_transform",
                _region(src, [0], 1, [16]),
                _region(dst, [0], 2, [16]),
            ),
            _async_call(
                "tl.dma_copy",
                _region(dst, [0], 1, [16]),
                _region(B, [k, 0], 2, [1, 16]),
                0,
            ),
        ]
    )
    loop = tir.For(
        k,
        0,
        8,
        tir.ForKind.SERIAL,
        body,
        annotations={"num_stages": tir.IntImm("int32", 2)},
    )
    root = tir.Block([], [], [], "root", loop, alloc_buffers=[src, dst])
    return tir.PrimFunc(
        [A.data, B.data],
        tir.BlockRealize([], True, root),
        buffer_map={A.data: A, B.data: B},
    )


@contextmanager
def _scoped_env(updates):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update({key: str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_ilp_models_layout_transform_destination_as_write(tmp_path):
    problem_path = tmp_path / "layout_transform_ilp_problem.json"
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = tvm.IRModule.from_expr(_make_layout_transform_pipeline().with_attr("global_symbol", "main"))

    with (
        tvm.target.Target(target),
        _scoped_env(
            {
                "TL_SUNMMIO_FASTER": "20",
                "TL_SUNMMIO_ILP_PROBLEM_JSON": problem_path,
            }
        ),
    ):
        tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)

    problem = json.loads(problem_path.read_text(encoding="utf-8"))
    assert problem["N"] == 3
    assert [1, 2] in problem["dep_edges"]
    assert problem["delta"]["1,2"] == 0
