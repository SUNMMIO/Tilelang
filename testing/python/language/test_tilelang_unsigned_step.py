"""Unsigned stepped-loop construction and execution without a GPU."""

import numpy as np
import pytest

import tilelang.language as T
from tilelang import tvm


def stepped_kernel(dtype="uint64", start=0, stop=3, step=(1 << 63) - 1, unroll=False, typed_step=True, read_bound=False, dynamic=False):
    start_value = tvm.tir.const(start, dtype)
    stop_value = tvm.tir.const(stop, dtype)
    step_value = tvm.tir.const(step, dtype) if typed_step and step is not None else step
    loop = T.unroll if unroll else T.serial

    @T.prim_func
    def main(output: T.Tensor((18,), "uint64"), bound: T.Tensor((1,), dtype), runtime_step: T.int32):
        output[0] = T.uint64(0)
        for i in loop(start_value, bound[0] if read_bound else stop_value, runtime_step if dynamic else step_value):
            output[0] += T.uint64(1)
            output[output[0]] = T.Cast("uint64", i)
            if dtype != "uint64":
                output[17] = T.Cast("uint64", i - 1 < 0)
            if read_bound:
                bound[0] = 0

    return main.with_attrs({"global_symbol": "main", "tir.noalias": True})


def parameter_bound_kernel():
    @T.prim_func
    def main(output: T.Tensor((1,), "uint64"), stop: T.uint64):
        output[0] = T.uint64(0)
        for _ in T.serial(T.uint64(0), stop, T.uint64(2)):
            output[0] += T.uint64(1)

    return main.with_attrs({"global_symbol": "main", "tir.noalias": True})


def uint8_range_grid_kernel(step):
    step_value = tvm.tir.const(step, "uint8") if step is not None else None

    @T.prim_func
    def main(
        starts: T.Tensor((65536,), "uint8"),
        stops: T.Tensor((65536,), "uint8"),
        counts: T.Tensor((65536,), "int32"),
        totals: T.Tensor((65536,), "int64"),
    ):
        for case in T.serial(65536):
            counts[case] = 0
            totals[case] = T.int64(0)
            for i in T.serial(starts[case], stops[case], step_value):
                counts[case] += 1
                totals[case] += T.Cast("int64", i)

    return main.with_attrs({"global_symbol": "main", "tir.noalias": True})


def signed_axis_kernel():
    @T.prim_func
    def main(output: T.Tensor((256,), "int32"), lo: T.int8, hi: T.int8):
        for i in T.serial(lo, hi):
            with T.block("emit"):
                vi = T.axis.remap("S", [i])
                output[T.Cast("int32", vi) + 128] = vi if vi.dtype == "int8" else 999

    return main.with_attrs({"global_symbol": "main", "tir.noalias": True})


def execute(func, dtype, stop, runtime_step=2):
    executable = tvm.compile(func, target="c").jit(options=["-std=c++17"])
    output = tvm.runtime.tensor(np.zeros(18, dtype="uint64"))
    bound = tvm.runtime.tensor(np.array([stop], dtype=dtype))
    executable(output, bound, runtime_step)
    data = output.numpy()
    return data[1 : 1 + int(data[0])].tolist(), data


@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize("dtype,step", [("uint8", 255), ("uint16", 65535), ("uint32", (1 << 32) - 1), ("uint64", (1 << 63) - 1)])
@pytest.mark.parametrize("typed_step", [False, True])
def test_large_unsigned_step_constructs_and_executes_once(dtype, step, unroll, typed_step):
    func = stepped_kernel(dtype=dtype, step=step, unroll=unroll, typed_step=typed_step)
    values, _ = execute(func, dtype, 3)
    assert values == [0]


@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize("stop,step", [(0, 7), (6, 7), (7, 7), (8, 7), (8, 2)])
def test_unsigned_step_empty_exact_and_remainder_counts(unroll, stop, step):
    func = stepped_kernel(stop=stop, step=step, unroll=unroll)
    values, _ = execute(func, "uint64", stop)
    assert values == list(range(0, stop, step))


@pytest.mark.parametrize("dtype,step", [("uint8", 255), ("uint16", 65535), ("uint32", (1 << 32) - 1)])
def test_unsigned_step_preserves_existing_index_promotion(dtype, step):
    func = stepped_kernel(dtype=dtype, stop=1, step=step)
    values, data = execute(func, dtype, 1)
    assert values == [0]
    assert data[17] == 1  # The promoted signed index makes i - 1 negative.


@pytest.mark.parametrize("unroll", [False, True])
def test_unsigned_runtime_bound_is_read_once_before_loop(unroll):
    func = stepped_kernel(step=2, unroll=unroll, read_bound=True)
    buffer = next(b for b in func.buffer_map.values() if b.name == "bound")
    reads = []
    tvm.tir.stmt_functor.post_order_visit(
        func.body, lambda n: reads.append(n) if isinstance(n, tvm.tir.BufferLoad) and n.buffer.same_as(buffer) else None
    )
    assert len(reads) == 1
    if unroll:
        # Runtime bounds can be constructed; unrolling requires a static extent.
        return
    values, _ = execute(func, "uint64", 3)
    assert values == [0, 2]


@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize("start,stop,step", [(0, 5, 2), (5, 0, -2)])
def test_signed_step_is_unchanged(unroll, start, stop, step):
    func = stepped_kernel(dtype="int32", start=start, stop=stop, step=step, unroll=unroll)
    values, _ = execute(func, "int32", stop)
    assert values == list(range(start, stop, step))


def test_zero_step_is_still_rejected():
    with pytest.raises(ValueError, match="step must be non-zero"):
        stepped_kernel(step=0)


def test_dynamic_step_keeps_existing_path():
    func = stepped_kernel(dtype="int32", step=2, dynamic=True)
    values, _ = execute(func, "int32", 3)
    assert values == [0, 2]
    values, _ = execute(func, "int32", 3, runtime_step=3)
    assert values == [0]


def test_python_step_larger_than_bound_dtype_keeps_promotion():
    func = stepped_kernel(dtype="uint8", step=256, typed_step=False)
    values, data = execute(func, "uint8", 3)
    assert values == [0]
    assert data[17] == 1


def test_scalar_bound_preserves_parameter_name_and_runtime_count():
    func = parameter_bound_kernel()
    assert func.params[1].name == "stop"
    executable = tvm.compile(func, target="c").jit(options=["-std=c++17"])
    for stop in (0, 1, 3):
        output = tvm.runtime.tensor(np.zeros(1, dtype="uint64"))
        executable(output, stop)
        assert int(output.numpy()[0]) == (stop + 1) // 2


@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize("dtype", ["uint8", "uint64"])
@pytest.mark.parametrize("step", [None, 1, 2])
def test_unsigned_literal_reverse_range_is_empty(unroll, dtype, step):
    func = stepped_kernel(dtype=dtype, start=5, stop=3, step=step, unroll=unroll)
    values, _ = execute(func, dtype, 3)
    assert values == []


@pytest.mark.parametrize("step", [None, 1, 2, 127, 255])
def test_all_uint8_runtime_bound_pairs_match_range_counts_and_values(step):
    func = uint8_range_grid_kernel(step)
    executable = tvm.compile(func, target="c").jit(options=["-std=c++17"])
    starts = np.repeat(np.arange(256, dtype="uint8"), 256)
    stops = np.tile(np.arange(256, dtype="uint8"), 256)
    counts = tvm.runtime.tensor(np.zeros(65536, dtype="int32"))
    totals = tvm.runtime.tensor(np.zeros(65536, dtype="int64"))
    executable(tvm.runtime.tensor(starts), tvm.runtime.tensor(stops), counts, totals)
    stride = step or 1
    distance = np.maximum(stops.astype("int64") - starts.astype("int64"), 0)
    expected = (distance + stride - 1) // stride
    sums = expected * starts + stride * expected * (expected - 1) // 2
    np.testing.assert_array_equal(counts.numpy(), expected)
    np.testing.assert_array_equal(totals.numpy(), sums)


def test_signed_range_preserves_remapped_axis_and_logical_iv_dtype():
    func = signed_axis_kernel()
    executable = tvm.compile(func, target="c").jit(options=["-std=c++17"])
    output = tvm.runtime.tensor(np.zeros(256, dtype="int32"))
    executable(output, -100, 100)
    expected = np.zeros(256, dtype="int32")
    expected[28:228] = np.arange(-100, 100, dtype="int32")
    np.testing.assert_array_equal(output.numpy(), expected)
