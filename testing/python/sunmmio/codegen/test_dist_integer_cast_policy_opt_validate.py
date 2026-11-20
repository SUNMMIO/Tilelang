"""Finite-width integer conversions from public kernels through SUVM codegen."""

import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    simplify_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
)
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls, collect_nodes, single_prim_func

tilelang.env.disable_cache()


@tilelang.jit(target="sunmmio")
def cast_quotient_kernel(flag=False, bias=-1, dtype="uint8", world_size: int = 1):
    kind = T.dist.SignalKind.SRAM_FLAGREG_INC if flag else T.dist.SignalKind.SRAM_MEMORY

    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            src = T.alloc_shared((512,), T.int32)
            dst = T.alloc_shared((256,), T.int32)
            signal = T.dist.signal(kind=kind, expect="manual")
            T.fill(src, 1)
            if dtype == "int8":
                q = T.Cast("int32", T.Cast("int8", n % 2 + 127) < 0)
            else:
                byte = T.Cast("int32", T.Cast(dtype, n % 2 + bias))
                q = T.truncdiv(byte, 128 if dtype == "uint8" else 32768)
            T.dist.put(src[q * 256 : q * 256 + 256], dst, (rank_id + 1) % world_size, signal=signal, submit=True)
            T.dist.wait_signal(signal, expected_delta=(256 * q if flag else q) + n % 2)

    return main


@tilelang.jit(target="sunmmio")
def wide_cast_kernel(source_type, dest_type, observation, world_size: int = 1):
    scalar_type = getattr(T, source_type)

    @T.prim_func
    def main(rank_id: T.dist.RankId, n: scalar_type):
        with T.Kernel():
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            value = T.Cast(dest_type, n)
            if observation == "negative":
                delta = T.Cast("uint32", value < 0)
            elif observation == "upper":
                delta = T.Cast("uint32", value >> T.uint64(32))
            else:
                delta = T.Cast("uint32", value)
            T.dist.wait_signal(signal, expected_delta=delta)

    return main


@tilelang.jit(target="sunmmio")
def storage_cast_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.uint64):
        with T.Kernel():
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            # The only narrowing here is inserted by the compiler for u32 storage.
            T.dist.wait_signal(signal, expected_delta=n)
            T.dist.wait_signal(signal, expected_delta=T.uint64(1))

    return main


@tilelang.jit(target="sunmmio")
def metadata_cast_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            buf = T.alloc_shared((T.Cast("int32", T.Cast("uint8", n)) + 256,), T.int32)
            T.fill(buf, 0)

    return main


@tilelang.jit(target="sunmmio")
def vector_cast_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            buf = T.alloc_shared((256,), T.uint8)
            buf[0:4] = T.Cast("uint8x4", T.Broadcast(n, 4))

    return main


@tilelang.jit(target="sunmmio")
def world_size_only_kernel(world_size: int = 1):
    @T.prim_func
    def main(out: T.Tensor((256,), T.int32), n: T.int32):
        with T.Kernel():
            local = T.alloc_shared((256,), T.int32)
            T.fill(local, n + T.dist.world_size())
            T.copy(local, out)

    return main


@tilelang.jit(target="sunmmio")
def frontend_cast_kernel(mode, strict=False, world_size: int = 1):
    @T.prim_func
    def main(A: T.MeshTensor((512,), T.placement.replicated(), T.int32), rank_id: T.dist.RankId, n: T.int32):
        with T.Kernel():
            q = T.truncdiv(T.Cast("int32", T.Cast("uint8", n % 2 - 1)), 128)
            if mode == "count":
                signals = T.dist.signals(1 + q, expect="manual")
                T.dist.wait_signal(signals[0], expected_delta=0)
            else:
                src = T.alloc_shared(((1 + q) * 256 if mode == "shape" else 256,), T.int32)
                dst = T.alloc_shared((256,), T.int32)
                signal = T.dist.signal(expect="manual")
                begin = q * 256
                T.copy(A[begin : begin + 256], src)
                T.dist.put(src, dst, (rank_id + 1) % world_size, signal=signal, submit=True)

    return main


def generate(kernel, tmp_path, **kwargs):
    result = lower_to_device_tir(kernel.get_tir(world_size=2, **kwargs))
    source = codegen_sunmmio_suvm_mlir(result.device_mod)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    return result, source


@pytest.mark.parametrize("dtype,bias", [("uint8", -1), ("uint8", 0), ("uint16", -1), ("int8", 0)])
def test_cast_quotient_survives_simplify_in_address_and_expected(tmp_path, dtype, bias):
    result, source = generate(cast_quotient_kernel, tmp_path, bias=bias, dtype=dtype)
    for n in (0, 1):
        bits = 16 if dtype == "uint16" else 8
        q = int(n == 1) if dtype == "int8" else ((n % 2 + bias) % (1 << bits)) // (1 << (bits - 1))
        batches, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [item[3] for batch in batches for item in batch if item[0] == "data"] == [256 * q]
        assert [event[1] for event in events if event[0] == "wait"] == [q + n % 2]


def test_cast_quotient_at_first_simplify():
    name = "tl.Simplify#1"
    result = lower_to_device_tir(cast_quotient_kernel.get_tir(world_size=2), capture_passes=name, capture_before_passes=name)
    for when in ("before", "after"):
        func = single_prim_func(result.pass_snapshot(name, when=when).mod)
        (put,) = collect_calls(func, "tl.tileop.dist_put")
        offset = put.args[0].indices[0].base
        lets = {node.var: node.value for node in collect_nodes(func, tvm.tir.LetStmt)}
        for _ in range(len(lets)):
            offset = tvm.tir.stmt_functor.substitute(offset, lets)
        n = next(v for v in func.params if v.name == "n")
        for value, expected in ((0, 256), (1, 0)):
            index = tvm.tir.stmt_functor.substitute(offset, {n: tvm.tir.const(value, "int32")})
            assert int(tvm.arith.Analyzer().simplify(index)) == expected


def test_cast_quotient_flag_peak_is_checked_before_narrowing():
    with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
        lower_to_device_tir(cast_quotient_kernel.get_tir(flag=True, world_size=2))


@pytest.mark.parametrize(
    "source_type,dest_type,observation,inputs",
    [
        ("int32", "uint64", "upper", [-(1 << 31), -1, 0, (1 << 31) - 1]),
        ("int64", "uint64", "upper", [-(1 << 63), -1, 0, (1 << 63) - 1]),
        ("uint64", "int64", "negative", [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]),
        ("uint64", "int8", "negative", [127, 128, (1 << 63) + 129, (1 << 64) - 1]),
        ("uint16", "int16", "low", [0, 32767, 32768, 65535]),
        ("int64", "int8", "low", [-(1 << 63), -129, -128, 127, 128, (1 << 63) - 1]),
        ("int64", "uint8", "low", [-(1 << 63), -(1 << 63) + 255, -1, 128, (1 << 63) - 1]),
        ("int64", "int32", "negative", [-(1 << 63), -1, (1 << 31) - 1, 1 << 31]),
        ("int64", "uint32", "low", [-(1 << 63), -1, 1 << 32, (1 << 63) - 1]),
        ("int32", "bool", "low", [-2, -1, 0, 1, 2]),
    ],
)
def test_finite_width_conversions_at_boundaries(tmp_path, source_type, dest_type, observation, inputs):
    result, source = generate(wide_cast_kernel, tmp_path, source_type=source_type, dest_type=dest_type, observation=observation)
    for n in inputs:
        if dest_type == "bool":
            expected = int(n != 0)
        else:
            width = int(dest_type.removeprefix("uint").removeprefix("int"))
            bits = n % (1 << width)
            if observation == "negative":
                expected = int(bits >= 1 << (width - 1))
            elif observation == "upper":
                expected = bits >> 32
            else:
                signed = dest_type.startswith("int") and bits >= 1 << (width - 1)
                expected = (bits - (1 << width) if signed else bits) % (1 << 32)
        _, events = trace_suvm_control_flow(result, source, n=n, rank_id=0)
        assert [event[1] for event in events if event[0] == "wait"] == [expected]


def test_memory_delta_generated_storage_cast(tmp_path):
    result, source = generate(storage_cast_kernel, tmp_path)
    for n in (0, 1 << 32, (1 << 32) + 3, (1 << 64) - 1):
        _, events = trace_suvm_control_flow(result, source, n=n, rank_id=0)
        assert [event[1] for event in events if event[0] == "wait"] == [n % (1 << 32), (n + 1) % (1 << 32)]


def test_unsupported_cast_fails_before_simplification():
    with pytest.raises(tvm.error.InternalError, match="cannot preserve integer Cast from int32 to int4"):
        lower_to_device_tir(wide_cast_kernel.get_tir("int32", "int4", "negative", world_size=2))


def test_wrapping_buffer_shape_is_rejected_at_entry():
    with pytest.raises(tvm.error.InternalError, match="Cast in buffer or annotation metadata"):
        lower_to_device_tir(metadata_cast_kernel.get_tir(world_size=2))


def test_unproven_vector_narrowing_diagnoses_the_conversion():
    with pytest.raises(tvm.error.InternalError, match="cannot preserve integer Cast from int32x4 to uint8x4"):
        lower_to_device_tir(vector_cast_kernel.get_tir(world_size=2))


def test_entry_policy_is_idempotent_and_preserves_lossless_casts():
    name = "tl.LegalizeSunmmioIntegerCasts"
    for bias in (-1, 0):
        kernel = cast_quotient_kernel.get_tir(bias=bias, world_size=2)
        result = lower_to_device_tir(kernel, capture_passes=name, capture_before_passes=name)
        original = result.pass_snapshot(name, when="before").mod
        normalized = result.pass_snapshot(name).mod
        assert tvm.ir.structural_equal(original, normalized) == (bias == 0)
        assert tvm.ir.structural_equal(normalized, tilelang.transform.LegalizeSunmmioIntegerCasts()(normalized))


def test_single_rank_skips_integer_cast_policy():
    mod = tvm.IRModule.from_expr(metadata_cast_kernel.get_tir(world_size=1))
    assert tvm.ir.structural_equal(mod, tilelang.transform.LegalizeSunmmioIntegerCasts()(mod))


def test_world_size_without_rank_id_still_compiles(tmp_path):
    result, _ = generate(world_size_only_kernel, tmp_path)
    func = single_prim_func(result.device_mod)
    assert int(func.attrs["tl.dist.world_size"]) == 2
    assert "tl.dist.rank_id_param_index" not in func.attrs


@pytest.mark.parametrize("strict", [True, False])
def test_frontend_cast_copy_dma_source(tmp_path, monkeypatch, strict):
    import tilelang.utils.target as target_utils

    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", strict)
    result, source = generate(frontend_cast_kernel, tmp_path, mode="copy", strict=strict)
    func = single_prim_func(result.device_mod)
    n_arg = next(i for i, var in enumerate(func.params) if var.name == "n")
    for n, partition in ((0, 1), (1, 0)):
        head, sep, body = source.partition(") {")
        body = re.sub(rf"%arg{n_arg}\b", "%fixed_n", body)
        fixed = simplify_suvm_mlir(
            f"{head}{sep}\n    %fixed_n = arith.constant {n} : i32\n{body}",
            opt_args=("--canonicalize", "--cse", "--sccp", "--canonicalize"),
        )
        dma_src = re.search(r"suvm.copy_async (%[\w-]+),", fixed)[1]
        index = re.search(rf"{dma_src} = suvm.get_partitioned_tile_view %arg\d+ indices = \[(%[\w-]+)\]", fixed)[1]
        assert re.search(rf"{index} = arith.constant {partition} : index", fixed)


@pytest.mark.parametrize("mode", ["shape", "count"])
def test_frontend_cast_metadata_is_not_a_static_constant(mode):
    with pytest.raises((ValueError, tvm.error.InternalError), match="buffer or annotation metadata|positive compile-time int"):
        frontend_cast_kernel.get_tir(mode=mode, world_size=2)
