import tilelang as tl
from tilelang import tvm
from tilelang import language as T
from tvm import tir


def _make_dangling_ping_pong_allocation():
    live = tir.decl_buffer((16,), "float32", name="live_ping", scope="shared.rsram")
    dead = tir.decl_buffer((16,), "float32", name="dead_pong", scope="shared.rsram")

    body = tir.BufferStore(live, tir.FloatImm("float32", 1), [0])
    body = tir.DeclBuffer(live, body)
    body = tir.Allocate(live.data, "float32", [16], True, body)
    body = tir.Allocate(
        dead.data,
        "float32",
        [16],
        True,
        body,
        annotations={"tl.sunmmio_alloc_ping_pong": "pong"},
    )

    layout = T.Layout((16,), lambda i: i)
    func = tir.PrimFunc([], body)
    func = func.with_attr("layout_map", {live: layout, dead: layout})
    func = func.with_attr("tl.sunmmio_alloc_ping_pong", {live.data: "ping", dead.data: "pong"})
    return tvm.IRModule.from_expr(func)


def _collect_buffer_declarations(func):
    allocations = set()
    declarations = set()

    def visit(node):
        if isinstance(node, tir.Allocate):
            allocations.add(node.buffer_var.name)
        elif isinstance(node, tir.DeclBuffer):
            declarations.add(node.buffer.name)

    tir.stmt_functor.post_order_visit(func.body, visit)
    return allocations, declarations


def test_remove_unused_sunmmio_allocations_cleans_dangling_metadata():
    result = tl.transform.RemoveUnusedSunmmioAllocations()(_make_dangling_ping_pong_allocation())["main"]

    allocations, declarations = _collect_buffer_declarations(result)
    assert allocations == {"live_ping"}
    assert declarations == {"live_ping"}

    layout_map = result.attrs["layout_map"]
    assert {buffer.name for buffer in layout_map} == {"live_ping"}

    ping_pong = result.attrs["tl.sunmmio_alloc_ping_pong"]
    assert {var.name for var in ping_pong} == {"live_ping"}


if __name__ == "__main__":
    test_remove_unused_sunmmio_allocations_cleans_dangling_metadata()
