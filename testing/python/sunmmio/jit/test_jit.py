import importlib.util
import json
import re
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.sunmmio import (
    SunmmioKernelABI,
    SunmmioKernelAdapter,
    SunmmioKernelSuDeckAdapter,
    SunmmioRuntimeScalar,
    SunmmioSuDeckLibraryGenerator,
    SunmmioSunsimLibraryGenerator,
)
from tilelang.jit.adapter.sunmmio import libgen as sunmmio_libgen
from tilelang.jit.adapter.sunmmio.libgen import (
    NpuirTools,
    SunmmioKernelArtifact,
    SunmmioToolchain,
)
from tilelang.jit.execution_backend import allowed_backends_for_target, resolve_execution_backend
from tilelang.layout import make_row_major, make_zz_layout
from tilelang.utils.target import determine_target, target_is_sunmmio


def _load_sunmmio_elementwise_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_add.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_elementwise_add_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sunmmio_dynamic_elementwise_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_add_dynamic.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_dynamic_elementwise_add_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sunmmio_dynamic_exp2_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_exp2_dynamic.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_dynamic_elementwise_exp2_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sunmmio_online_softmax_example():
    example_path = Path(__file__).resolve().parents[3] / "examples" / "sunmmio" / "softmax" / "online_softmax.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_online_softmax_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_npuir_tools():
    try:
        return NpuirTools.resolve()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_sunmmio_toolchain():
    try:
        return SunmmioToolchain.resolve()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_sunmmio_codegen():
    build = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("Sunmmio SUVM codegen is not available in this build. Rebuild TileLang with USE_SUNMMIO=ON.")
    return build


def _runtime_scalar_sources(abi):
    return [(scalar.name, scalar.source_param_index, scalar.source_kind, scalar.source_dim) for scalar in abi.runtime_scalars]


def test_sunmmio_abi_reorders_public_args_to_device_order():
    abi = SunmmioKernelABI(
        kernel_name="main_kernel",
        public_arg_count=3,
        public_param_names=("X", "LSE", "Y"),
        device_param_names=("LSE", "X", "Y"),
        runtime_scalars=(),
    )

    assert abi.materialize_runtime_args(["x", "lse", "y"], lambda *_: None) == ["lse", "x", "y"]


def test_sunmmio_abi_reorders_public_args_before_hidden_scalars():
    class Marker:
        def __init__(self, shape):
            self.shape = shape

    abi = SunmmioKernelABI(
        kernel_name="main_kernel",
        public_arg_count=3,
        public_param_names=("X", "LSE", "Y"),
        device_param_names=("LSE", "X", "Y", "m"),
        runtime_scalars=(SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),),
    )
    x = Marker((512, 128))
    lse = Marker((512, 128))
    y = Marker((512, 128))

    runtime_args = abi.materialize_runtime_args(
        [x, lse, y],
        lambda marker, source_kind, dim_index: marker.shape[dim_index] if source_kind == "shape" else None,
    )

    assert runtime_args == [lse, x, y, 512]
    assert abi.materialize_runtime_args([x, lse, y, 512], lambda *_: None) == [lse, x, y, 512]


def test_sunmmio_abi_rejects_public_arg_count_mismatch():
    with pytest.raises(ValueError, match="public_arg_count"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=3,
            public_param_names=("X", "Y"),
            device_param_names=("X", "Y"),
            runtime_scalars=(),
        )


def test_sunmmio_abi_rejects_duplicate_public_names():
    with pytest.raises(ValueError, match="duplicate public"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=2,
            public_param_names=("X", "X"),
            device_param_names=("X", "X"),
            runtime_scalars=(),
        )


def test_sunmmio_abi_rejects_duplicate_scalar_names():
    with pytest.raises(ValueError, match="duplicate scalar"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=1,
            public_param_names=("X",),
            device_param_names=("X", "m"),
            runtime_scalars=(
                SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),
                SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=1),
            ),
        )


def test_sunmmio_abi_rejects_public_scalar_name_collision():
    with pytest.raises(ValueError, match="collide"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=1,
            public_param_names=("m",),
            device_param_names=("m",),
            runtime_scalars=(SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),),
        )


def test_sunmmio_abi_rejects_non_permutation_device_order():
    with pytest.raises(ValueError, match="not a permutation"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=2,
            public_param_names=("X", "Y"),
            device_param_names=("X", "Z"),
            runtime_scalars=(),
        )


def test_sunmmio_runtime_scalar_rejects_partial_source():
    with pytest.raises(ValueError, match="partial source binding"):
        SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape")


def test_sunmmio_abi_from_json_dict_rejects_inconsistent_metadata():
    with pytest.raises(ValueError, match="not a permutation"):
        SunmmioKernelABI.from_json_dict(
            {
                "kernel_name": "k",
                "public_arg_count": 2,
                "public_param_names": ["X", "Y"],
                "device_param_names": ["X", "Z"],
                "runtime_scalars": [],
            }
        )


def test_sunmmio_abi_from_modules_preserves_public_and_device_orders():
    x_buffer = tvm.tir.decl_buffer((128, 128), "bfloat16", name="X")
    lse_buffer = tvm.tir.decl_buffer((128, 128), "float32", name="LSE")
    y_buffer = tvm.tir.decl_buffer((128, 128), "float32", name="Y")
    prim_func = tvm.tir.PrimFunc(
        [x_buffer.data, lse_buffer.data, y_buffer.data],
        tvm.tir.Evaluate(0),
        buffer_map={
            x_buffer.data: x_buffer,
            lse_buffer.data: lse_buffer,
            y_buffer.data: y_buffer,
        },
    ).with_attr("global_symbol", "main")
    params = [KernelParam.from_buffer(buffer) for buffer in (x_buffer, lse_buffer, y_buffer)]

    device_func = tvm.tir.PrimFunc(
        [
            tvm.tir.Var("LSE", "handle"),
            tvm.tir.Var("X", "handle"),
            tvm.tir.Var("Y", "handle"),
        ],
        tvm.tir.Evaluate(0),
    ).with_attr("tir.is_global_func", True)
    device_mod = tvm.IRModule({"main_kernel": device_func})

    abi = SunmmioKernelABI.from_modules(
        func_or_mod=prim_func,
        host_mod=None,
        device_mod=device_mod,
        params=params,
    )

    assert abi.kernel_name == "main_kernel"
    assert abi.public_param_names == ("X", "LSE", "Y")
    assert abi.device_param_names == ("LSE", "X", "Y")
    assert abi.materialize_runtime_args(["x", "lse", "y"], lambda *_: None) == ["lse", "x", "y"]


def test_sunmmio_backend_resolution_accepts_lowercase_target():
    target = determine_target("sunmmio", return_object=True)

    assert target_is_sunmmio(target)
    assert allowed_backends_for_target(target) == ["sunmmio", "sunmmio_sunsim"]
    assert resolve_execution_backend(None, target) == "sunmmio"
    assert resolve_execution_backend("auto", target) == "sunmmio"
    assert resolve_execution_backend("Sunmmio", target) == "sunmmio"
    assert resolve_execution_backend("sunmmio_sunsim", target) == "sunmmio_sunsim"


def test_sunmmio_backends_share_cache_class():
    from tilelang.cache import _dispatch_map
    from tilelang.jit.adapter.sunmmio.kernel_cache import SunmmioKernelCache

    assert type(_dispatch_map["sunmmio"]) is SunmmioKernelCache
    assert type(_dispatch_map["sunmmio_sunsim"]) is SunmmioKernelCache


def test_sunmmio_toolchain_resolves_only_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SUNMMIO_TOOLCHAIN", raising=False)

    with pytest.raises(FileNotFoundError, match="SUNMMIO_TOOLCHAIN is not set"):
        SunmmioToolchain.resolve()

    toolchain_root = tmp_path / "toolchain"
    clangxx = toolchain_root / "clang" / "bin" / "clang++"
    clangxx.parent.mkdir(parents=True)
    clangxx.write_text("#!/bin/sh\n", encoding="utf-8")
    clangxx.chmod(0o755)
    monkeypatch.setenv("SUNMMIO_TOOLCHAIN", str(toolchain_root))

    toolchain = SunmmioToolchain.resolve()

    assert toolchain.clangxx == clangxx
    assert toolchain.cflags()[0] == "--target=riscv64-sunmmio-elf"
    assert all(not flag.startswith("--sysroot=") for flag in toolchain.cflags())


def test_sunmmio_adapter_resolves_target_before_libgen(tmp_path, monkeypatch):
    prim_func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("global_symbol", "kernel")
    observed_targets = []

    original_init = SunmmioSuDeckLibraryGenerator.__init__

    def record_init(self, target, kernel_name, verbose=False):
        observed_targets.append(target)
        original_init(self, target, kernel_name, verbose)

    monkeypatch.setattr(SunmmioSuDeckLibraryGenerator, "__init__", record_init)
    monkeypatch.setattr(SunmmioSuDeckLibraryGenerator, "compile_lib", lambda self: None)
    monkeypatch.setattr(SunmmioSuDeckLibraryGenerator, "load_lib", lambda self, lib_path=None: None)

    adapter = SunmmioKernelSuDeckAdapter(
        params=[],
        result_idx=[],
        target="llvm -mcpu=sunmmio-test -mattr=device_mesh_nrow_4,device_mesh_ncol_4",
        func_or_mod=prim_func,
        device_kernel_source="",
    )
    toolchain = SunmmioToolchain(clangxx=tmp_path / "clang++")
    cflags = adapter.lib_generator._compile_flags(toolchain)

    assert len(observed_targets) == 1
    assert target_is_sunmmio(observed_targets[0])
    assert str(observed_targets[0].attrs["mcpu"]) == "sunmmio-test"
    assert str(adapter.target.attrs["mcpu"]) == "sunmmio-test"
    assert [flag for flag in cflags if flag.startswith("-mcpu=")] == ["-mcpu=sunmmio-test"]


def test_sunmmio_cache_persists_artifacts_through_libgen(tmp_path):
    from tilelang.jit.adapter.sunmmio.kernel_cache import SunmmioKernelCache

    class Kernel:
        pass

    class Adapter:
        pass

    target = determine_target("sunmmio", return_object=True)

    kernel = Kernel()
    kernel.execution_backend = "sunmmio"
    kernel.adapter = Adapter()
    kernel.adapter.lib_generator = SunmmioSuDeckLibraryGenerator(target, "kernel")
    src_sudeck_elf = tmp_path / "source-sudeck.elf"
    src_sudeck_elf.write_bytes(b"SUDECK_ELF")
    kernel.adapter.lib_generator.artifact = SunmmioKernelArtifact(
        elf_path=str(src_sudeck_elf),
        llvm_ir_source="define void @kernel() { ret void }",
        llvm_ir_path=str(tmp_path / "source-sudeck.ll"),
        build_dir=str(tmp_path),
        runtime_kernel_name="kernel",
    )

    sunmmio_cache_dir = tmp_path / "sunmmio"
    sunmmio_cache_dir.mkdir()
    SunmmioKernelCache()._save_so_cubin_to_disk(kernel, str(sunmmio_cache_dir))

    kernel_elf = sunmmio_cache_dir / "kernel.elf"
    kernel_ll = sunmmio_cache_dir / "kernel.ll"
    assert kernel_elf.read_bytes() == b"SUDECK_ELF"
    assert kernel_ll.read_text(encoding="utf-8") == "define void @kernel() { ret void }"
    assert json.loads((sunmmio_cache_dir / "abi.json").read_text(encoding="utf-8")) == {"kernel_name": "kernel"}
    assert kernel.adapter.lib_generator.artifact.llvm_ir_path == str(kernel_ll)
    assert kernel.adapter.lib_generator.artifact.llvm_ir_source == "define void @kernel() { ret void }"
    assert kernel.adapter.lib_generator.libpath == str(kernel_elf)

    src_elf = tmp_path / "source.elf"
    src_elf.write_bytes(b"ELF")
    kernel.execution_backend = "sunmmio_sunsim"
    kernel.adapter.lib_generator = SunmmioSunsimLibraryGenerator(target, "kernel")
    kernel.adapter.lib_generator.artifact = SunmmioKernelArtifact(
        elf_path=str(src_elf),
        llvm_ir_source="define void @kernel() { ret void }",
        llvm_ir_path=str(tmp_path / "source.ll"),
        build_dir=str(tmp_path),
        runtime_kernel_name="kernel",
    )

    sunsim_cache_dir = tmp_path / "sunsim"
    sunsim_cache_dir.mkdir()
    SunmmioKernelCache()._save_so_cubin_to_disk(kernel, str(sunsim_cache_dir))

    assert (sunsim_cache_dir / "kernel.elf").read_bytes() == b"ELF"
    assert (sunsim_cache_dir / "kernel.ll").read_text(encoding="utf-8") == "define void @kernel() { ret void }"
    assert json.loads((sunsim_cache_dir / "abi.json").read_text(encoding="utf-8")) == {"kernel_name": "kernel"}
    artifact = kernel.adapter.lib_generator.artifact
    assert artifact.elf_path == str(sunsim_cache_dir / "kernel.elf")
    assert artifact.llvm_ir_path == str(sunsim_cache_dir / "kernel.ll")
    assert artifact.llvm_ir_source == "define void @kernel() { ret void }"


def test_sunmmio_sudeck_compile_emits_kernel_elf_without_thunk(tmp_path, monkeypatch):
    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSuDeckLibraryGenerator(target, "kernel")
    llvm_source = "define void @kernel(ptr %0) #0 !sunmmio.kernel_meta !1 { ret void }"

    fake_toolchain = SunmmioToolchain(clangxx=tmp_path / "clang++")
    commands = []

    def fake_resolve():
        return fake_toolchain

    def fake_run_command(command, **kwargs):
        commands.append([str(part) for part in command])
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_bytes(b"ELF")

        class Result:
            stdout = ""

        return Result()

    def fake_materialize_llvm_ir(output_path=None, **kwargs):
        Path(output_path).write_text(llvm_source, encoding="utf-8")
        return llvm_source

    monkeypatch.setattr(sunmmio_libgen.SunmmioToolchain, "resolve", fake_resolve)
    monkeypatch.setattr(sunmmio_libgen, "_run_command", fake_run_command)
    monkeypatch.setattr(generator, "materialize_llvm_ir", fake_materialize_llvm_ir)

    generator.compile_lib(output_dir=tmp_path)

    assert (tmp_path / "kernel.ll").read_text(encoding="utf-8") == generator.artifact.llvm_ir_source
    assert (tmp_path / "kernel.elf").read_bytes() == b"ELF"
    assert not (tmp_path / "main_thunk.cpp").exists()
    assert not (tmp_path / "main_thunk.o").exists()
    assert len(commands) == 2
    assert commands[0][:2] == [str(fake_toolchain.clangxx), "-c"]
    assert "--target=riscv64-sunmmio-elf" in commands[0]
    assert all(not flag.startswith("--sysroot=") for command in commands for flag in command)
    assert str(tmp_path / "kernel.ll") in commands[0]
    assert str(tmp_path / "kernel.o") in commands[0]
    assert commands[1][0] == str(fake_toolchain.clangxx)
    assert "-c" not in commands[1]
    assert str(tmp_path / "kernel.o") in commands[1]
    assert str(tmp_path / "kernel.elf") in commands[1]
    assert "-nostartfiles" in commands[1]
    assert "-lnosys" in commands[1]
    for command in commands:
        assert "main_thunk.o" not in command


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_sunsim_elementwise_add_example_executes(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    pytest.importorskip("ml_dtypes")
    pytest.importorskip("sunsim")

    from tilelang.cache import _dispatch_map

    example = _load_sunmmio_elementwise_example()
    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        example.elementwise_add._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()

        shapes = [(64, 64), (256, 64)]
        for shape in shapes:
            result = example.main(*shape)
            assert result.exit_code == 0

        elf_files = list(cache_root.glob("*/kernel.elf"))

        assert len(elf_files) == len(shapes)
    finally:
        example.elementwise_add._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_sunsim_dynamic_elementwise_add_example_executes(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    pytest.importorskip("ml_dtypes")
    pytest.importorskip("sunsim")

    from tilelang.cache import _dispatch_map

    example = _load_sunmmio_dynamic_elementwise_example()
    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        example.elementwise_add_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()

        shapes = [(64, 64), (96, 160), (129, 257)]
        for shape in shapes:
            result = example.main(*shape)
            assert result.exit_code == 0

        elf_files = list(cache_root.glob("*/kernel.elf"))

        assert len(elf_files) == 1
    finally:
        example.elementwise_add_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_sunsim_dynamic_elementwise_exp2_example_executes(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    pytest.importorskip("sunsim")

    from tilelang.cache import _dispatch_map

    example = _load_sunmmio_dynamic_exp2_example()
    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        example.elementwise_exp2_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()

        shapes = [(64, 64), (96, 160), (129, 257)]
        for shape in shapes:
            result = example.main(*shape)
            assert result.exit_code == 0

        elf_files = list(cache_root.glob("*/kernel.elf"))

        assert len(elf_files) == 1
    finally:
        example.elementwise_exp2_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


def test_sunmmio_sunsim_dynamic_elementwise_add_jit_binds_shape_abi():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    host_source = artifact.host_mod.script()
    device_source = artifact.kernel_source

    assert example.elementwise_add_dynamic.execution_backend == "sunmmio_sunsim"
    assert len(artifact.params) == 3
    assert all(not param.is_scalar() for param in artifact.params)
    assert 'with T.LetStmt(T.Cast("int32", elem_add_A_shape_1[0]), var=m)' in host_source
    assert 'with T.LetStmt(T.Cast("int32", elem_add_A_shape_1[1]), var=n)' in host_source
    assert 'T.call_extern("int32", "elem_add_kernel", A.data, B.data, C.data, m, n)' in host_source
    assert "func.func @elem_add_kernel" in device_source
    assert "%arg3: i32" in device_source
    assert "%arg4: i32" in device_source


def test_sunmmio_sunsim_dynamic_elementwise_exp2_jit_binds_shape_abi():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_exp2_example()
    prim_func = example.elementwise_exp2_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.float32,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    host_source = artifact.host_mod.script()
    device_source = artifact.kernel_source

    assert example.elementwise_exp2_dynamic.execution_backend == "sunmmio_sunsim"
    assert len(artifact.params) == 2
    assert all(not param.is_scalar() for param in artifact.params)
    assert 'with T.LetStmt(T.Cast("int32", elem_exp2_A_shape_1[0]), var=m)' in host_source
    assert 'with T.LetStmt(T.Cast("int32", elem_exp2_A_shape_1[1]), var=n)' in host_source
    assert 'T.call_extern("int32", "elem_exp2_kernel", A.data, B.data, m, n)' in host_source
    assert "func.func @elem_exp2_kernel" in device_source
    assert "%arg2: i32" in device_source
    assert "%arg3: i32" in device_source


def test_sunmmio_abi_extracts_dynamic_shape_bindings():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    abi = SunmmioKernelABI.from_modules(
        func_or_mod=prim_func,
        host_mod=artifact.host_mod,
        device_mod=artifact.device_mod,
        params=artifact.params,
    )

    assert abi.kernel_name == "elem_add_kernel"
    assert abi.public_arg_count == 3
    assert abi.public_param_names == ("A", "B", "C")
    assert abi.runtime_scalar_names == ("m", "n")
    assert _runtime_scalar_sources(abi) == [
        ("m", 0, "shape", 0),
        ("n", 0, "shape", 1),
    ]

    class Marker:
        shape = (129, 257)

    args = [Marker(), Marker(), Marker()]
    runtime_args = abi.materialize_runtime_args(
        args,
        lambda marker, source_kind, dim_index: marker.shape[dim_index] if source_kind == "shape" else None,
    )

    assert runtime_args[:3] == args
    assert runtime_args[3:] == [129, 257]
    assert abi.materialize_runtime_args([*args, 129, 257], lambda *_: None) == [*args, 129, 257]


def test_sunmmio_abi_falls_back_to_prim_func_for_cached_dynamic_kernel():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    abi = SunmmioKernelABI.from_modules(
        func_or_mod=prim_func,
        host_mod=None,
        device_mod=None,
        params=artifact.params,
    )

    assert abi.kernel_name == "elem_add"
    assert abi.public_arg_count == 3
    assert abi.public_param_names == ("A", "B", "C")
    assert abi.runtime_scalar_names == ("m", "n")
    assert _runtime_scalar_sources(abi) == [
        ("m", 0, "shape", 0),
        ("n", 0, "shape", 1),
    ]


def test_sunmmio_sunsim_cache_restores_abi_kernel_name_from_metadata(tmp_path):
    from tilelang.jit.adapter.sunmmio.kernel_cache import SunmmioKernelCache

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    params = [KernelParam.from_buffer(prim_func.buffer_map[param]) for param in prim_func.params]

    elf_path = tmp_path / "kernel.elf"
    ll_path = tmp_path / "kernel.ll"
    elf_path.write_bytes(b"ELF")
    ll_path.write_text(
        "define void @elem_add_kernel(ptr %0, ptr %1, ptr %2, i32 %3, i32 %4) #0 !sunmmio.kernel_meta !1 { ret void }",
        encoding="utf-8",
    )
    cached_abi = SunmmioKernelABI(
        kernel_name="elem_add_kernel",
        public_arg_count=3,
        public_param_names=("A", "B", "C"),
        device_param_names=("B", "A", "C", "m", "n"),
        runtime_scalars=(
            SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),
            SunmmioRuntimeScalar("n", source_param_index=0, source_kind="shape", source_dim=1),
        ),
    )
    (tmp_path / "abi.json").write_text(json.dumps(cached_abi.to_json_dict()), encoding="utf-8")

    kernel = SunmmioKernelCache()._build_kernel(
        func=prim_func,
        host_kernel_source=None,
        device_kernel_source="module {}",
        kernel_lib_path=str(elf_path),
        kernel_params=params,
        target="sunmmio",
        target_host=None,
        out_idx=[],
        execution_backend="sunmmio_sunsim",
        pass_configs=None,
        compile_flags=None,
    )
    assert kernel is not None
    adapter = kernel.adapter

    assert adapter.abi.kernel_name == "elem_add_kernel"
    assert adapter.runtime_kernel_name == "elem_add_kernel"
    assert adapter.abi.public_param_names == ("A", "B", "C")
    assert adapter.abi.device_param_names == ("B", "A", "C", "m", "n")
    assert adapter.abi.runtime_scalar_names == ("m", "n")
    assert _runtime_scalar_sources(adapter.abi) == [
        ("m", 0, "shape", 0),
        ("n", 0, "shape", 1),
    ]


def test_sunmmio_elementwise_sync_waits_are_not_duplicated():
    _require_sunmmio_codegen()

    static_example = _load_sunmmio_elementwise_example()
    static_prim = static_example.elementwise_add.get_tir(
        64,
        64,
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    static_artifact = tilelang.lower(
        static_prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    static_source = static_artifact.device_mod.script()

    assert static_source.count("T.dma_copy") == 3
    assert static_source.count("T.wait_token(0)") == 1
    assert static_source.count("T.wait_token(1)") == 1
    assert static_source.count("T.wait_token(2)") == 1

    static_multi_prim = static_example.elementwise_add.get_tir(
        256,
        64,
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    static_multi_artifact = tilelang.lower(
        static_multi_prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    static_multi_source = static_multi_artifact.device_mod.script()

    assert static_multi_source.count("T.dma_copy") == 3
    assert static_multi_source.count("T.wait_token(0)") == 1
    assert static_multi_source.count("T.wait_token(1)") == 1
    assert static_multi_source.count("T.wait_token(2)") == 2
    assert "T.sync_null_token(2)" in static_multi_source

    dynamic_example = _load_sunmmio_dynamic_elementwise_example()
    dynamic_prim = dynamic_example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    dynamic_artifact = tilelang.lower(
        dynamic_prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    dynamic_source = dynamic_artifact.device_mod.script()

    assert dynamic_source.count("T.dma_copy") == 3
    assert dynamic_source.count("T.wait_token(0)") == 1
    assert dynamic_source.count("T.wait_token(1)") == 1
    assert dynamic_source.count("T.wait_token(2)") == 2
    assert "T.sync_null_token(2)" in dynamic_source


def test_sunmmio_softmax_output_dma_wait_is_hoisted_before_tile_store_loop():
    _require_sunmmio_codegen()

    example = _load_sunmmio_online_softmax_example()
    prim = example.online_softmax.get_tir(
        2048,
        2048,
        block_M=256,
        block_N=128,
        dtype=T.bfloat16,
    )
    artifact = tilelang.lower(
        prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    lines = artifact.device_mod.script().splitlines()

    output_dma_idx, output_dma_line = next(
        (idx, line) for idx, line in enumerate(lines) if "T.dma_copy(T.region(Y_shared[0, 0]" in line and "T.sync_token_id(" in line
    )
    output_token = int(re.search(r"T\.sync_token_id\((\d+)\)", output_dma_line).group(1))
    wait_marker = f"T.wait_token({output_token})"
    null_marker = f"T.sync_null_token({output_token})"

    store_idx = max(idx for idx, line in enumerate(lines[:output_dma_idx]) if "Y_shared[" in line and "] =" in line)
    input_dma_idx = max(
        idx for idx, line in enumerate(lines[:store_idx]) if "T.dma_copy(T.region(X_" in line and "T.sync_token_id(" in line
    )
    tile_store_loop_idx = next(
        idx
        for idx, line in enumerate(lines[input_dma_idx:store_idx], start=input_dma_idx)
        if "for i in T.serial" in line and "tile.domain" in line
    )
    outer_loop_idx = max(idx for idx, line in enumerate(lines[:input_dma_idx]) if "for bx_1 in range" in line)

    assert all(null_marker not in line for line in lines[outer_loop_idx:output_dma_idx])
    assert any(wait_marker in line for line in lines[input_dma_idx:tile_store_loop_idx])
    assert all(wait_marker not in line for line in lines[tile_store_loop_idx:output_dma_idx])


def test_sunmmio_base_adapter_does_not_expose_sunsim_runtime_surface(tmp_path):
    assert not hasattr(SunmmioKernelAdapter, "get_llvm_ir")
    assert not hasattr(SunmmioKernelAdapter, "lower_to_llvm_ir")
    assert not hasattr(SunmmioKernelAdapter, "libpath")
    assert not hasattr(SunmmioKernelAdapter, "llvm_ir_artifact")
    assert not hasattr(SunmmioKernelAdapter, "llvm_ir_source")
    assert not hasattr(SunmmioKernelAdapter, "llvm_ir_path")
    assert not hasattr(SunmmioKernelAdapter, "build_sunsim_elf")
    assert not hasattr(SunmmioKernelAdapter, "sunsim_artifact")
    assert not hasattr(SunmmioKernelAdapter, "sunsim_elf_path")
    assert not hasattr(SunmmioKernelAdapter, "sunsim_build_dir")

    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSunsimLibraryGenerator(target, "kernel")
    llvm_path = tmp_path / "kernel.ll"

    generator.artifact = SunmmioKernelArtifact(
        elf_path=str(tmp_path / "kernel.elf"),
        llvm_ir_source="define void @kernel() { ret void }",
        llvm_ir_path=str(llvm_path),
        build_dir=str(tmp_path),
        runtime_kernel_name="kernel",
    )

    assert generator.artifact.llvm_ir_path == str(llvm_path)
    assert generator.get_lib_path() is None


def test_sunmmio_sunsim_thunk_uses_abi_kernel_name(tmp_path, monkeypatch):
    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSunsimLibraryGenerator(target, "elem_add_kernel")
    generator.update_mlir_source(
        """
module attributes {suvm.device_arch = #suvm.device_arch<a4e>} {
  func.func @elem_add_kernel() {
    return
  }
}
"""
    )
    llvm_source = """
define void @elem_add_kernel(ptr addrspace(4) %0) #0 !sunmmio.kernel_meta !1 {
  ret void
}
"""

    def fake_materialize_llvm_ir(output_path=None, **kwargs):
        Path(output_path).write_text(llvm_source, encoding="utf-8")
        return llvm_source

    fake_toolchain = SunmmioToolchain(clangxx=tmp_path / "clang++")
    commands = []

    def fake_resolve():
        return fake_toolchain

    def fake_run_command(command, **kwargs):
        commands.append([str(part) for part in command])
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_bytes(b"artifact")

        class Result:
            stdout = ""

        return Result()

    monkeypatch.setattr(sunmmio_libgen.SunmmioToolchain, "resolve", fake_resolve)
    monkeypatch.setattr(sunmmio_libgen, "_run_command", fake_run_command)
    monkeypatch.setattr(generator, "materialize_llvm_ir", fake_materialize_llvm_ir)

    generator.compile_lib(output_dir=tmp_path)
    thunk = (tmp_path / "main_thunk.cpp").read_text(encoding="utf-8")

    assert generator.runtime_kernel_name == "elem_add_kernel"
    assert "void elem_add_kernel(void *args);" in thunk
    assert "elem_add_kernel(const_cast<unsigned char *>(_kernel_arg_start));" in thunk
    assert '#include "pwln_setup.h"' not in thunk
    assert 'section(".r.sram.common")' in thunk
    assert "static inline void pwln_init()" in thunk
    assert len(commands) == 3
    assert all("--target=riscv64-sunmmio-elf" in command for command in commands)
    assert all(not flag.startswith("--sysroot=") for command in commands for flag in command)


@tilelang.jit(target="sunmmio", out_idx=[2])
def elementwise_add_jit(M, N, block_M, block_N, in_dtype, out_dtype):
    """JIT version of examples/sunmmio/elementwise/elementwise_add.py."""

    zz_layout = make_zz_layout((M, N))
    placement = T.MeshShardingPolicy(y=0, x=1)

    @T.prim_func
    def elem_add(
        A: T.MeshTensor((M, N), placement, in_dtype, layout=zz_layout),
        B: T.MeshTensor((M, N), placement, in_dtype, layout=zz_layout),
        C: T.MeshTensor((M, N), placement, out_dtype, layout=zz_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_N = A.local_shape

            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(A[bx * block_M, by * block_N], A_shared)
                    T.copy(B[bx * block_M, by * block_N], B_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        C_shared[i, j] = A_shared[i, j] + B_shared[i, j]
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return elem_add


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_jit_lowercase_target_generates_suvm_source():
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    kernel = elementwise_add_jit(64, 64, 32, 32, T.bfloat16, T.float32)
    source = kernel.get_kernel_source()

    assert kernel.execution_backend == "sunmmio"
    assert isinstance(kernel.adapter, SunmmioKernelSuDeckAdapter)
    assert target_is_sunmmio(kernel.target)
    assert "suvm.device_arch" in source
    assert "func.func @elem_add" in source

    with pytest.raises(RuntimeError, match="Sunmmio SuDeck JIT runtime execution is not implemented yet"):
        kernel(None, None)


@tilelang.jit(target="sunmmio")
def row_major_copy_jit(M, N, dtype):
    rsram_layout = make_row_major((M, N))

    @T.prim_func
    def copy_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1):
            A_shared = T.alloc_shared((M, N), dtype, scope="shared.rsram")
            T.annotate_layout({A_shared: rsram_layout})

            T.copy(A[0, 0], A_shared)
            T.copy(A_shared, B[0, 0])

    return copy_kernel


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_jit_row_major_copy_lowers_to_llvm_ir(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    kernel = row_major_copy_jit(32, 64, T.float32)
    llvm_path = tmp_path / "row_major_copy_jit.ll"

    llvm_ir = kernel.adapter.lib_generator.materialize_llvm_ir(
        output_path=llvm_path,
    )

    assert llvm_path.read_text(encoding="utf-8") == llvm_ir
    assert "suvm.device_arch" in kernel.get_kernel_source()
    assert "define void @copy_kernel" in llvm_ir
    assert "sunmmio.kernel_meta" in llvm_ir
    assert "su_odma_submit_direct" in llvm_ir


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_jit_cache_saves_llvm_ir(tmp_path, monkeypatch):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    from tilelang.cache import _dispatch_map

    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        row_major_copy_jit._kernel_cache.clear()
        _dispatch_map["sunmmio"]._memory_cache.clear()

        kernel = row_major_copy_jit(32, 64, T.float32)
        elf_files = sorted(cache_root.glob("*/kernel.elf"))
        ll_files = sorted(cache_root.glob("*/kernel.ll"))

        assert len(elf_files) == 1
        assert len(ll_files) == 1
        assert elf_files[0].read_bytes()
        llvm_ir = ll_files[0].read_text(encoding="utf-8")
        artifact = kernel.adapter.lib_generator.artifact
        assert artifact is not None
        assert artifact.llvm_ir_source == llvm_ir
        assert Path(artifact.llvm_ir_path) == ll_files[0]
        assert Path(artifact.elf_path) == elf_files[0]
        assert "define void @copy_kernel" in llvm_ir
        assert "sunmmio.kernel_meta" in llvm_ir

        def fail_resolve(*args, **kwargs):
            raise AssertionError("cached Sunmmio LLVM IR should not require NPU-IR tool resolution")

        monkeypatch.setattr(NpuirTools, "resolve", fail_resolve)
        row_major_copy_jit._kernel_cache.clear()
        _dispatch_map["sunmmio"]._memory_cache.clear()
        cached_kernel = row_major_copy_jit(32, 64, T.float32)

        cached_artifact = cached_kernel.adapter.lib_generator.artifact
        assert cached_artifact is not None
        assert cached_artifact.llvm_ir_source == llvm_ir
        assert Path(cached_artifact.llvm_ir_path) == ll_files[0]
        assert Path(cached_artifact.elf_path) == elf_files[0]
    finally:
        row_major_copy_jit._kernel_cache.clear()
        _dispatch_map["sunmmio"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


if __name__ == "__main__":
    tilelang.testing.main()
