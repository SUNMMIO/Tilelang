from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import tilelang
from tilelang import tvm
from tilelang.jit.adapter.sunmmio import SunmmioKernelABI
from tilelang.jit.adapter.sunmmio import suvm_edit_session as session_module
from tilelang.jit.adapter.sunmmio import adapter as adapter_module
from tilelang.jit.adapter.sunmmio.adapter import (
    SunmmioKernelSuDeckAdapter,
    SunmmioSunsimKernelAdapter,
)
from tilelang.jit.adapter.sunmmio.libgen import SunmmioKernelArtifact
from tilelang.jit.adapter.sunmmio.suvm_edit_session import SunmmioSuvmEditSession


def _empty_kernel():
    return tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("global_symbol", "main")


def _empty_abi():
    return SunmmioKernelABI(
        kernel_name="main_kernel",
        public_arg_count=0,
        public_param_names=(),
        device_param_names=(),
        device_param_dtypes=(),
        runtime_scalars=(),
    )


def _fake_lowered(source: str):
    device_func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("tir.is_global_func", True)
    return SimpleNamespace(
        kernel_source=source,
        host_mod=None,
        device_mod=tvm.IRModule({"main_kernel": device_func}),
        params=[],
    )


def _write_compile_inputs(session):
    artifacts = session.artifacts
    artifacts.path(session_module.ORIGINAL_MLIR).write_text("module {}\n", encoding="utf-8")
    artifacts.edited_mlir.write_text("module { // edited\n}\n", encoding="utf-8")
    artifacts.path(session_module.DEVICE_TIR).write_text("# device tir\n", encoding="utf-8")
    session_module._write_json(artifacts.path(session_module.ABI_FILE), _empty_abi().to_json_dict())
    session_module._write_json(
        artifacts.path(session_module.MANIFEST_FILE),
        {
            "schema_version": session_module.MANIFEST_SCHEMA_VERSION,
            "target": "sunmmio",
            "opt_level": 3,
            "parameters": [],
        },
    )


def test_suvm_edit_session_repeated_emit_archives_previous_edit(tmp_path, monkeypatch):
    lowered = _fake_lowered("module { // fresh\n}\n")
    monkeypatch.setattr(tilelang, "lower", lambda *_args, **_kwargs: lowered)
    monkeypatch.setattr(
        session_module.SunmmioKernelABI,
        "from_modules",
        lambda **_kwargs: _empty_abi(),
    )

    session = SunmmioSuvmEditSession(tmp_path)
    session.emit(_empty_kernel())
    session.artifacts.edited_mlir.write_text("module { // manual edit\n}\n", encoding="utf-8")
    session.emit(_empty_kernel())

    archives = list(tmp_path.glob("kernel.edited.*Z.mlir"))
    assert len(archives) == 1
    assert "manual edit" in archives[0].read_text(encoding="utf-8")
    assert session.artifacts.edited_mlir.read_text(encoding="utf-8") == lowered.kernel_source


def test_suvm_edit_session_lowering_failure_keeps_previous_edit(tmp_path, monkeypatch):
    edited = tmp_path / "kernel.edited.mlir"
    edited.write_text("module { // keep me\n}\n", encoding="utf-8")
    monkeypatch.setattr(
        tilelang,
        "lower",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("lowering failed")),
    )

    with pytest.raises(RuntimeError, match="lowering failed"):
        SunmmioSuvmEditSession(tmp_path).emit(_empty_kernel())

    assert "keep me" in edited.read_text(encoding="utf-8")
    assert list(tmp_path.glob("kernel.edited.*Z.mlir")) == []


def test_suvm_edit_session_compile_sunsim_uses_edited_mlir(tmp_path, monkeypatch):
    session = SunmmioSuvmEditSession(tmp_path, timeout=17.0)
    _write_compile_inputs(session)

    captured = {}

    class FakeGenerator:
        def __init__(self, target, kernel_name, verbose=False):
            captured.setdefault("kernel_names", []).append(kernel_name)
            self.artifact = None

        def update_mlir_source(self, source):
            captured["mlir"] = source

        def update_device_tir_source(self, source):
            captured["tir"] = source

        def compile_lib(self, timeout, output_dir):
            captured["timeout"] = timeout
            elf = Path(output_dir) / "kernel.elf"
            mlir = Path(output_dir) / "kernel.mlir"
            llvm = Path(output_dir) / "kernel.ll"
            elf.write_bytes(b"ELF")
            mlir.write_text(captured["mlir"], encoding="utf-8")
            llvm.write_text("define void @main_kernel() {}\n", encoding="utf-8")
            self.artifact = SunmmioKernelArtifact(
                elf_path=elf,
                mlir_path=mlir,
                llvm_ir_path=llvm,
                build_dir=Path(output_dir),
                runtime_kernel_name="main_kernel",
            )

        def load_lib(self, kernel_lib_path):
            captured["loaded"] = Path(kernel_lib_path)
            output_dir = Path(kernel_lib_path).parent
            self.artifact = SunmmioKernelArtifact(
                elf_path=Path(kernel_lib_path),
                mlir_path=output_dir / "kernel.mlir",
                llvm_ir_path=output_dir / "kernel.ll",
                build_dir=output_dir,
                runtime_kernel_name="main_kernel",
            )

    monkeypatch.setattr(SunmmioSuvmEditSession, "_validate_mlir", lambda *_args: None)
    monkeypatch.setattr(session_module, "SunmmioSunsimLibraryGenerator", FakeGenerator)
    monkeypatch.setattr(adapter_module, "SunmmioSunsimLibraryGenerator", FakeGenerator)
    monkeypatch.setattr(session_module, "_validate_sunsim_elf_abi", lambda *_args: None)

    executable = session.compile_sunsim()

    assert isinstance(executable, SunmmioSunsimKernelAdapter)
    assert captured == {
        "kernel_names": ["main_kernel", "main_kernel"],
        "mlir": "module { // edited\n}\n",
        "tir": "# device tir\n",
        "timeout": 17.0,
        "loaded": tmp_path / "kernel.elf",
    }
    assert executable._artifact_parameter_kinds == ()
    assert not hasattr(executable, "_artifact_timeout")
    assert executable.lib_generator.artifact.elf_path == tmp_path / "kernel.elf"

    runs = []
    fake_sunsim = SimpleNamespace(
        Input=type("Input", (), {}),
        Output=type("Output", (), {}),
        Inout=type("Inout", (), {}),
        Descriptor=type("Descriptor", (), {}),
        run=lambda **kwargs: runs.append(kwargs),
    )
    monkeypatch.setattr(
        SunmmioSunsimKernelAdapter,
        "_import_sunsim",
        staticmethod(lambda: fake_sunsim),
    )
    executable(timeout=31.0)
    assert runs == [
        {
            "elf": tmp_path / "kernel.elf",
            "args": [],
            "kernel_name": "main_kernel",
            "timeout": 31.0,
        }
    ]


def test_suvm_edit_session_compile_uses_sudeck_runtime(tmp_path, monkeypatch):
    session = SunmmioSuvmEditSession(tmp_path, timeout=19.0)
    _write_compile_inputs(session)
    captured = {}

    class FakeGenerator:
        def __init__(self, target, kernel_name, verbose=False):
            captured.setdefault("kernel_names", []).append(kernel_name)
            self.artifact = None
            self.pymodule = None

        def update_launcher_specs(self, specs):
            captured.setdefault("launcher_specs", []).append(specs)

        def update_mlir_source(self, source):
            captured["mlir"] = source

        def update_device_tir_source(self, source):
            captured["tir"] = source

        def compile_lib(self, timeout, output_dir):
            captured["timeout"] = timeout
            output_dir = Path(output_dir)
            (output_dir / "kernel.elf").write_bytes(b"ELF")
            self.artifact = SunmmioKernelArtifact(
                elf_path=output_dir / "kernel.elf",
                mlir_path=output_dir / "kernel.mlir",
                llvm_ir_path=output_dir / "kernel.ll",
                build_dir=output_dir,
                runtime_kernel_name="main_kernel",
            )

        def load_lib(self, kernel_lib_path):
            captured["loaded"] = Path(kernel_lib_path)
            output_dir = Path(kernel_lib_path).parent
            self.artifact = SunmmioKernelArtifact(
                elf_path=Path(kernel_lib_path),
                mlir_path=output_dir / "kernel.mlir",
                llvm_ir_path=output_dir / "kernel.ll",
                build_dir=output_dir,
                runtime_kernel_name="main_kernel",
            )
            self.pymodule = SimpleNamespace(call=lambda *_args: None)

    monkeypatch.setattr(SunmmioSuvmEditSession, "_validate_mlir", lambda *_args: None)
    monkeypatch.setattr(session_module, "SunmmioSuDeckLibraryGenerator", FakeGenerator)
    monkeypatch.setattr(adapter_module, "SunmmioSuDeckLibraryGenerator", FakeGenerator)

    executable = session.compile()

    assert isinstance(executable, SunmmioKernelSuDeckAdapter)
    assert captured == {
        "kernel_names": ["main_kernel", "main_kernel"],
        "launcher_specs": [[], []],
        "mlir": "module { // edited\n}\n",
        "tir": "# device tir\n",
        "timeout": 19.0,
        "loaded": tmp_path / "kernel.elf",
    }
    assert not hasattr(executable, "_artifact_parameter_kinds")
    assert not hasattr(executable, "_artifact_timeout")
    assert executable.lib_generator.artifact.elf_path == tmp_path / "kernel.elf"


def test_sudeck_adapter_loads_artifact_and_launches_torch_sunmmio_handles(
    tmp_path,
    monkeypatch,
):
    stream = object()
    torch_module = ModuleType("torch")
    torch_module.sunmmio = SimpleNamespace(
        current_stream=lambda: stream,
        unsafe_get_sudeck_stream_handle=lambda value: 0x1234 if value is stream else 0,
    )
    torch_sunmmio_module = ModuleType("torch_sunmmio")
    torch_sunmmio_runtime = ModuleType("torch_sunmmio.sunmmio")
    tensor = SimpleNamespace(shape=(8, 16), stride=lambda: (16, 1))
    torch_sunmmio_runtime.unsafe_get_sutensor_handle = lambda value: 0x5678 if value is tensor else 0
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch_sunmmio", torch_sunmmio_module)
    monkeypatch.setitem(sys.modules, "torch_sunmmio.sunmmio", torch_sunmmio_runtime)

    calls = []
    abi = SunmmioKernelABI(
        kernel_name="main_kernel",
        public_arg_count=1,
        public_param_names=("A",),
        device_param_names=("A",),
        device_param_dtypes=("handle",),
        runtime_scalars=(),
    )

    class FakeGenerator:
        def __init__(self, target, kernel_name, verbose=False):
            self.artifact = None
            self.pymodule = SimpleNamespace(call=lambda *args: calls.append(args))

        def update_launcher_specs(self, specs):
            assert specs == [("A", "tensor")]

        def load_lib(self, kernel_lib_path):
            self.artifact = SunmmioKernelArtifact(
                elf_path=Path(kernel_lib_path),
                mlir_path=tmp_path / "kernel.mlir",
                llvm_ir_path=tmp_path / "kernel.ll",
                build_dir=tmp_path,
                runtime_kernel_name="main_kernel",
            )

    elf_path = tmp_path / "kernel.elf"
    elf_path.write_bytes(b"ELF")
    monkeypatch.setattr(adapter_module, "SunmmioSuDeckLibraryGenerator", FakeGenerator)
    executable = SunmmioKernelSuDeckAdapter.from_compiled_artifact(
        target="sunmmio",
        abi=abi,
        kernel_lib_path=elf_path,
    )

    executable(tensor)

    assert calls == [(0x5678, 0x1234)]
