from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _buffer_names(attr_map):
    return {var.name for var in attr_map}


def test_hoist_layout_annotations_to_func_attrs():
    @T.prim_func
    def before(A: T.Buffer((16,), "float32")):
        B = T.alloc_buffer((16,), "float32", scope="shared")
        C = T.alloc_buffer((16,), "float32", scope="global")
        with T.block("producer"):
            T.block_attr({"layout_map": {B.data: T.int32(1)}})
            T.evaluate(0)
        with T.block("consumer"):
            T.block_attr(
                {
                    "layout_map": {A.data: T.int32(2)},
                    "global_layout_map": {C.data: T.int32(3)},
                }
            )
            T.evaluate(0)

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.HoistBlockAnnotationsToFuncAttrs()(mod)
    func = mod["main"]

    assert "layout_map" in func.attrs
    assert "global_layout_map" in func.attrs
    assert _buffer_names(func.attrs["layout_map"]) == {"A", "B"}
    assert _buffer_names(func.attrs["global_layout_map"]) == {"C"}

    layout_values = {var.name: int(value) for var, value in func.attrs["layout_map"].items()}
    global_layout_values = {var.name: int(value) for var, value in func.attrs["global_layout_map"].items()}
    assert layout_values == {"A": 2, "B": 1}
    assert global_layout_values == {"C": 3}


if __name__ == "__main__":
    tilelang.testing.main()
