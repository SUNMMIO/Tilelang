from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tvm.target import Target

from tilelang.env import TL_BINS
from tilelang.jit.adapter.libgen import LibraryGenerator


SUNMMIO_TOOLCHAIN_ENV = "SUNMMIO_TOOLCHAIN"

DEFAULT_NPUIR_OPT_ARGS = (
    "--verify-each",
    "-suvm-assign-dma-channels",
    "-suvm-resolve-tokens",
    "-suvm-emit-kernel-abi",
    "-convert-suvm-to-llvm",
    "-canonicalize",
    "-cse",
    "-convert-scf-to-cf",
    "-convert-arith-to-llvm",
    "-convert-cf-to-llvm",
    "-reconcile-unrealized-casts",
)

SUNMMIO_CLANG_CFLAGS = (
    "--target=riscv64-sunmmio-elf",
    "-O2",
)
SUNMMIO_LDFLAGS = (
    "-fuse-ld=lld",
    "-Wl,--no-warn-mismatch,--gc-sections",
    "-lnosys",
)
SUNMMIO_SUDECK_LDFLAGS = (
    "-nostartfiles",
    *SUNMMIO_LDFLAGS,
)
_SUNSIM_PWLN_TABLE_WORDS = (
    0x3FB551D5,
    0x3FBD580D,
    0x3FC5BA53,
    0x3FCE7B54,
    0x3FD79FA0,
    0x3FE12B85,
    0x3FEB239E,
    0x3FF58CB4,
    0x400035E4,
    0x4005E30B,
    0x400BD086,
    0x40120137,
    0x401877F6,
    0x401F385F,
    0x402643CD,
    0x402DA4B2,
    0x3F1547C6,
    0x3F043A8F,
    0x3EE2BAE1,
    0x3EB9261F,
    0x3E8B70A0,
    0x3E32A462,
    0x3D89F69D,
    0xBD4AF4CE,
    0xBE35322A,
    0xBEA18D78,
    0xBEEE9CB5,
    0xBF211709,
    0xBF4E5641,
    0xBF7F493C,
    0xBF9A0F79,
    0xBFB6A6F0,
    0xBF70EC02,
    0xBF56282E,
    0xBF3FA069,
    0xBF2C7082,
    0xBF1C00DC,
    0xBF0DEAFD,
    0xBF01856E,
    0xBEED78C8,
    0xBEDA7BE2,
    0xBEC9AB51,
    0xBEBAB44F,
    0xBEAD693F,
    0xBEA181B3,
    0xBE96A8CA,
    0xBE8CFEFE,
    0xBE83CCC7,
    0x3FF8675B,
    0x3FEA3186,
    0x3FDD86F4,
    0x3FD22403,
    0x3FC7DF3B,
    0x3FBEA220,
    0x3FB61D4E,
    0x3FAE5D82,
    0x3FA73F79,
    0x3FA0AE8D,
    0x3F9A9A9A,
    0x3F94FF70,
    0x3F8FCABC,
    0x3F8AE0D9,
    0x3F8659A7,
    0x3F81E549,
    0xBEF46B51,
    0xBEDFA893,
    0xBECDD108,
    0xBEBE3051,
    0xBEB073F6,
    0xBEA44A9E,
    0xBE99774C,
    0xBE8FC15F,
    0xBE870D20,
    0xBE7E84B3,
    0xBE70403D,
    0xBE6325BA,
    0xBE571825,
    0xBE4BFD2D,
    0xBE41BD2C,
    0xBE384294,
    0x3FBD123B,
    0x3FB78BD1,
    0x3FB27F8F,
    0x3FADE03B,
    0x3FA9A059,
    0x3FA5B48D,
    0x3FA21204,
    0x3F9EB15A,
    0x3F9B8B97,
    0x3F989AF8,
    0x3F95DA3F,
    0x3F93450A,
    0x3F90D806,
    0x3F8E90A0,
    0x3F8C6C5B,
    0x3F8A68ED,
    0x40784DF0,
    0x406A0FCD,
    0x405D5FCD,
    0x405208CD,
    0x4047DDCD,
    0x403EB98D,
    0x40367D8D,
    0x402F0B8D,
    0x40284F8D,
    0x4022355D,
    0x401CAD5D,
    0x4017A8DD,
    0x40131A3D,
    0x400EF62D,
    0x400B316D,
    0x4007C20D,
    0xC016C5D8,
    0xC012F74C,
    0xC00F6948,
    0xC00C1190,
    0xC008E7B0,
    0xC005E510,
    0xC003049C,
    0xC0004380,
    0xBFFA0758,
    0xBFF5180C,
    0xBFF05898,
    0xBFEBC0E8,
    0xBFE74E58,
    0xBFE2FF7C,
    0xBFDED288,
    0xBFDAC634,
)
_C_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PathLike = str | os.PathLike[str]


@dataclass(frozen=True)
class SunmmioKernelArtifact:
    elf_path: str
    llvm_ir_source: str
    llvm_ir_path: str
    build_dir: str
    runtime_kernel_name: str


def _find_existing_path(
    description: str,
    candidates: Sequence[Path],
    *,
    executable: bool = False,
) -> Path:
    for candidate in candidates:
        if candidate.exists() and (not executable or os.access(candidate, os.X_OK)):
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{description} not found. Checked:\n{checked}")


@dataclass(frozen=True)
class NpuirTools:
    opt: Path
    translate: Path

    @classmethod
    def resolve(cls) -> NpuirTools:
        return cls(
            opt=cls._find_tool("npuir-opt"),
            translate=cls._find_tool("npuir-translate"),
        )

    @staticmethod
    def _find_tool(name: str) -> Path:
        candidates = [Path(bin_dir) / name for bin_dir in TL_BINS]

        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        checked = "\n".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(
            f"{name} executable not found in TileLang binary directories. Rebuild TileLang with USE_SUNMMIO=ON "
            f"or install a Sunmmio-enabled TileLang package that includes NPU-IR tools. Checked:\n{checked}"
        )


@dataclass(frozen=True)
class SunmmioToolchain:
    clangxx: Path

    @classmethod
    def resolve(cls) -> SunmmioToolchain:
        toolchain_env = os.getenv(SUNMMIO_TOOLCHAIN_ENV)
        if not toolchain_env:
            raise FileNotFoundError(f"{SUNMMIO_TOOLCHAIN_ENV} is not set. Set it to the Sunmmio toolchain root.")

        toolchain_path = _find_existing_path("Sunmmio toolchain", [Path(toolchain_env)])
        return cls(
            clangxx=_find_existing_path(
                "Sunmmio clang++",
                [toolchain_path / "clang" / "bin" / "clang++"],
                executable=True,
            )
        )

    def cflags(self) -> list[str]:
        return list(SUNMMIO_CLANG_CFLAGS)


def _sanitize_llvm_ir_for_sunmmio_toolchain(llvm_ir: str) -> str:
    # The current zpu clang is LLVM 17 and does not accept LLVM-19+ GEP hint flags.
    return llvm_ir.replace("getelementptr inbounds nuw", "getelementptr inbounds").replace("getelementptr nuw", "getelementptr")


def _split_compile_flags(compile_flags: list[str] | None, existing: Sequence[str]) -> list[str]:
    if not compile_flags:
        return []
    existing_set = set(existing)
    return [item for flag in compile_flags for item in flag.split() if item not in existing_set]


def _target_mcpu(target: Target) -> str:
    mcpu = target.attrs.get("mcpu") if target.attrs is not None else None
    if mcpu is None:
        raise ValueError(f"Sunmmio target is missing required `mcpu` attribute: {target}")
    return str(mcpu)


def _run_command(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    description: str,
    input_text: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed\ncommand: {' '.join(str(part) for part in command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _render_sunsim_pwln_setup() -> str:
    table_rows = []
    for offset in range(0, len(_SUNSIM_PWLN_TABLE_WORDS), 8):
        row = ", ".join(f"0x{word:08X}" for word in _SUNSIM_PWLN_TABLE_WORDS[offset : offset + 8])
        table_rows.append(f"    {row},")
    table = "\n".join(table_rows)
    return f"""\
__attribute__((section(".r.sram.common"))) static const uint32_t pwln_tables[{len(_SUNSIM_PWLN_TABLE_WORDS)}] = {{
{table}
}};

__device static inline void pwln_init() {{
  __rsram float *base = (__rsram float *)pwln_tables;
  auto t_exp = su_vload(base);
  auto t_recip = su_vload(base + 32);
  auto t_rsqrt = su_vload(base + 64);
  auto t_ln = su_vload(base + 96);
  (void)su_vfpwln_set_exp(t_exp);
  (void)su_vfpwln_set_recip(t_recip);
  (void)su_vfpwln_set_rsqrt(t_rsqrt);
  (void)su_vfpwln_set_ln(t_ln);
  su_sync_vector();
}}
"""


def _render_sunsim_main_thunk(kernel_name: str) -> str:
    if not _C_IDENTIFIER_RE.match(kernel_name):
        raise ValueError(f"Sunmmio kernel name must be a C identifier for sunsim main thunk generation: {kernel_name!r}")

    return f"""\
#include <stddef.h>
#include <stdint.h>

#include <sunmmio_dev.h>

{_render_sunsim_pwln_setup()}

extern "C" {{

__attribute__((used, section(".kernargs"), aligned(8))) volatile unsigned char _kernel_arg_start[4096] = {{1}};
__attribute__((used, section(".descriptors"), aligned(8))) volatile unsigned char _descriptor_start[4096] = {{1}};

void {kernel_name}(void *args);

int main(void) {{
  pwln_init();
  {kernel_name}(const_cast<unsigned char *>(_kernel_arg_start));
  return 0;
}}

}}  // extern "C"
    """


class SunmmioLibraryGenerator(LibraryGenerator):
    """Generate Sunmmio library artifacts from TileLang SUVM MLIR."""

    def __init__(self, target: Target, verbose: bool = False):
        super().__init__(target, verbose)
        self.mlir_source: str = ""
        self.artifact: SunmmioKernelArtifact | None = None

    def update_mlir_source(self, mlir_source: str):
        self.mlir_source = mlir_source

    def _compile_flags(self, toolchain: SunmmioToolchain) -> list[str]:
        cflags = toolchain.cflags()
        cflags.append(f"-mcpu={_target_mcpu(self.target)}")
        cflags += _split_compile_flags(self.compile_flags, cflags)
        return cflags

    def load_lib(self, lib_path: PathLike | None = None):
        if lib_path is None:
            if self.libpath is None:
                raise RuntimeError("SunmmioLibraryGenerator.libpath is not set; call compile_lib() first or pass lib_path explicitly.")
            lib_path = self.libpath
        elf_path = Path(lib_path)
        if elf_path.suffix != ".elf":
            raise ValueError(f"Sunmmio kernel artifact must be an ELF file, got: {elf_path}")

        llvm_ir_path = elf_path.parent / "kernel.ll"
        if not llvm_ir_path.exists():
            raise RuntimeError(f"Cached Sunmmio ELF is missing sibling LLVM IR artifact: {elf_path}")

        llvm_ir_source = llvm_ir_path.read_text(encoding="utf-8")
        runtime_kernel_name = getattr(self, "runtime_kernel_name", "kernel")
        self.artifact = SunmmioKernelArtifact(
            elf_path=str(elf_path),
            llvm_ir_source=llvm_ir_source,
            llvm_ir_path=str(llvm_ir_path),
            build_dir=str(elf_path.parent),
            runtime_kernel_name=runtime_kernel_name,
        )
        self.srcpath = str(llvm_ir_path)
        self.libpath = str(elf_path)
        if hasattr(self, "runtime_kernel_name"):
            self.runtime_kernel_name = runtime_kernel_name

    def materialize_llvm_ir(
        self,
        output_path: PathLike | None = None,
        *,
        opt_args: Sequence[str] = DEFAULT_NPUIR_OPT_ARGS,
    ) -> str:
        """Lower the generated SUVM MLIR source to LLVM IR with NPU-IR tools."""

        if not self.mlir_source.strip():
            raise ValueError("Sunmmio kernel has no SUVM MLIR source to lower")

        tools = NpuirTools.resolve()
        output = Path(output_path) if output_path is not None else None

        with tempfile.TemporaryDirectory(prefix="tilelang-sunmmio-") as tmp_dir:
            input_path = Path(tmp_dir) / "kernel.mlir"
            input_path.write_text(self.mlir_source, encoding="utf-8")
            llvm_dialect = self._run_npuir_opt(tools, input_path, opt_args)
            llvm_ir = self._run_npuir_translate(tools, llvm_dialect, output)
            llvm_ir = _sanitize_llvm_ir_for_sunmmio_toolchain(llvm_ir)
            if output is not None:
                output.write_text(llvm_ir, encoding="utf-8")
            return llvm_ir

    def _compile_kernel_obj(
        self,
        toolchain: SunmmioToolchain,
        llvm_path: Path,
        build_dir: Path,
        timeout: float | None,
    ) -> Path:
        kernel_obj = build_dir / "kernel.o"
        _run_command(
            [toolchain.clangxx, "-c", *self._compile_flags(toolchain), "-o", kernel_obj, llvm_path],
            description="Sunmmio kernel LLVM IR compilation",
            timeout=timeout,
        )
        return kernel_obj

    def _run_npuir_opt(self, tools: NpuirTools, input_path: Path, opt_args: Sequence[str]) -> str:
        command = [str(tools.opt), str(input_path), *opt_args]
        return _run_command(
            command,
            description=f"npuir-opt failed while lowering Sunmmio SUVM MLIR\nmlir: {input_path}",
        ).stdout

    def _run_npuir_translate(self, tools: NpuirTools, llvm_dialect: str, output: Path | None) -> str:
        command = [str(tools.translate), "--mlir-to-llvmir"]
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            command.extend(["-o", str(output)])

        result = _run_command(
            command,
            input_text=llvm_dialect,
            description="npuir-translate failed while translating Sunmmio LLVM dialect",
        )
        return output.read_text(encoding="utf-8") if output is not None else result.stdout

    def compile_lib(self, timeout: float | None = None):
        raise NotImplementedError("SunmmioLibraryGenerator only lowers to LLVM IR; use a runtime-specific generator.")


class SunmmioSuDeckLibraryGenerator(SunmmioLibraryGenerator):
    """Build a SuDeck-loadable Sunmmio kernel ELF from LLVM IR."""

    def __init__(self, target: Target, kernel_name: str, verbose: bool = False):
        super().__init__(target, verbose)
        self.kernel_name = kernel_name
        self.runtime_kernel_name = kernel_name

    def compile_lib(
        self,
        timeout: float | None = None,
        output_dir: PathLike | None = None,
    ):
        build_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="tilelang-sunmmio-sudeck-"))
        build_dir.mkdir(parents=True, exist_ok=True)
        sunmmio_toolchain = SunmmioToolchain.resolve()

        llvm_path = build_dir / "kernel.ll"
        llvm_ir = self.materialize_llvm_ir(output_path=llvm_path)
        kernel_obj = self._compile_kernel_obj(sunmmio_toolchain, llvm_path, build_dir, timeout)
        elf_path = build_dir / "kernel.elf"
        _run_command(
            [sunmmio_toolchain.clangxx, *self._compile_flags(sunmmio_toolchain), kernel_obj, *SUNMMIO_SUDECK_LDFLAGS, "-o", elf_path],
            description="Sunmmio SuDeck kernel ELF link",
            timeout=timeout,
        )

        self.artifact = SunmmioKernelArtifact(
            elf_path=str(elf_path),
            llvm_ir_source=llvm_ir,
            llvm_ir_path=str(llvm_path),
            build_dir=str(build_dir),
            runtime_kernel_name=self.runtime_kernel_name,
        )
        self.srcpath = str(llvm_path)
        self.libpath = str(elf_path)


class SunmmioSunsimLibraryGenerator(SunmmioLibraryGenerator):
    """Build a sunsim-loadable ELF from Sunmmio LLVM IR."""

    def __init__(self, target: Target, kernel_name: str, verbose: bool = False):
        super().__init__(target, verbose)
        self.kernel_name = kernel_name
        self.runtime_kernel_name = kernel_name

    def compile_lib(
        self,
        timeout: float | None = None,
        output_dir: PathLike | None = None,
    ):
        build_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="tilelang-sunmmio-sunsim-"))
        build_dir.mkdir(parents=True, exist_ok=True)
        sunmmio_toolchain = SunmmioToolchain.resolve()

        llvm_path = build_dir / "kernel.ll"
        llvm_ir = self.materialize_llvm_ir(output_path=llvm_path)
        thunk_path = self._write_main_thunk(build_dir)
        kernel_obj = self._compile_kernel_obj(sunmmio_toolchain, llvm_path, build_dir, timeout)
        thunk_obj = self._compile_thunk_obj(sunmmio_toolchain, thunk_path, build_dir, timeout)
        elf_path = self._link_elf(sunmmio_toolchain, kernel_obj, thunk_obj, build_dir, timeout)

        self.artifact = SunmmioKernelArtifact(
            elf_path=str(elf_path),
            llvm_ir_source=llvm_ir,
            llvm_ir_path=str(llvm_path),
            build_dir=str(build_dir),
            runtime_kernel_name=self.runtime_kernel_name,
        )
        self.srcpath = str(llvm_path)
        self.libpath = str(elf_path)

    def _write_main_thunk(self, build_dir: Path) -> Path:
        thunk_path = build_dir / "main_thunk.cpp"
        thunk_path.write_text(_render_sunsim_main_thunk(self.runtime_kernel_name), encoding="utf-8")
        return thunk_path

    def _compile_thunk_obj(
        self,
        toolchain: SunmmioToolchain,
        thunk_path: Path,
        build_dir: Path,
        timeout: float | None,
    ) -> Path:
        thunk_obj = build_dir / "main_thunk.o"
        _run_command(
            [
                toolchain.clangxx,
                "-c",
                *self._compile_flags(toolchain),
                "-x",
                "sunmmio",
                "-o",
                thunk_obj,
                thunk_path,
            ],
            description="Sunmmio sunsim main thunk compilation",
            timeout=timeout,
        )
        return thunk_obj

    def _link_elf(
        self,
        toolchain: SunmmioToolchain,
        kernel_obj: Path,
        thunk_obj: Path,
        build_dir: Path,
        timeout: float | None,
    ) -> Path:
        elf_path = build_dir / "kernel.elf"
        _run_command(
            [toolchain.clangxx, *self._compile_flags(toolchain), kernel_obj, thunk_obj, *SUNMMIO_LDFLAGS, "-o", elf_path],
            description="Sunmmio sunsim ELF link",
            timeout=timeout,
        )
        return elf_path
