"""Sunmmio SuDeck Wrapper for

This module provides C++ kernel launcher generation for Sunmmio
- Automatic C++ launcher generation with SuDeck API
"""

from __future__ import annotations

from dataclasses import dataclass

_TENSOR = "tensor"

_DTYPE_TO_C = {
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
}


@dataclass(frozen=True)
class SunmmioSuDeckParam:
    """One device-ABI argument: a tensor handle or a scalar of a given TVM dtype."""

    name: str
    kind: str  # "tensor" or a TVM scalar dtype (e.g. "int32")

    @property
    def is_tensor(self) -> bool:
        return self.kind == _TENSOR

    @property
    def c_type(self) -> str:
        if self.is_tensor:
            return "int64_t"
        try:
            return _DTYPE_TO_C[self.kind]
        except KeyError:
            raise ValueError(f"SuDeck launcher: unsupported scalar dtype {self.kind!r} for param {self.name!r}") from None


# TODO: How can we cache the kernel elf without reloading it every time
# if we hit the kernel cache
_LAUNCHER_TEMPLATE = """\
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <sudeck/context.h>
#include <sudeck/error.h>
#include <sudeck/program.h>
#include <sudeck/stream.h>
#include <sudeck/tensor.h>

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

using namespace sudeck;

namespace {{

// Unwrap a sudeck Result or throw; tvm-ffi surfaces the throw as a Python exception.
template <typename T>
T check(Result<T> result, const char *what) {{
  if (result) return std::move(*result);
  std::ostringstream oss;
  oss << "launch_kernel " << what << ": " << result.error().message() << " (code=" << result.error().code() << ")";
  throw std::runtime_error(oss.str());
}}

void check(Result<void> result, const char *what) {{
  if (result) return;
  std::ostringstream oss;
  oss << "launch_kernel " << what << ": " << result.error().message() << " (code=" << result.error().code() << ")";
  throw std::runtime_error(oss.str());
}}

}}  // namespace

void launch_kernel({signature_params}) {{
  KernelSpec spec = check(KernelSpec::from_elf(std::string_view(elf.data(), elf.size()),
                                               std::string_view(name.data(), name.size())),
                          "from_elf");
  if (stream_handle == 0) {{
    throw std::runtime_error("launch_kernel stream: null sudeck stream handle");
  }}
  const Stream *stream_ptr = reinterpret_cast<const Stream *>(static_cast<uintptr_t>(stream_handle));
  Stream stream = *stream_ptr;
  if (!stream) {{
    throw std::runtime_error("launch_kernel stream: invalid sudeck stream handle");
  }}
  std::vector<KernelArg> args;
  args.reserve({nargs});
{arg_pushes}
  check(stream.launch(spec, args), "launch");  // submit only; torch-sunmmio owns stream synchronization
}}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_kernel, launch_kernel);
"""

_PY_MODULE_TEMPLATE = '''\
"""Generated SuDeck launch module for a single TileLang kernel."""
import os

import tvm_ffi

_module_dir = os.path.dirname(os.path.abspath(__file__))
_ELF = os.path.join(_module_dir, {elf_name!r}).encode("utf-8")
_NAME = {kernel_name!r}.encode("utf-8")
_launcher = None


def _load():
    global _launcher
    if _launcher is None:
        _launcher = tvm_ffi.load_module(os.path.join(_module_dir, {launcher_lib_name!r}))["launch_kernel"]
    return _launcher


def call({call_params}):
    _load()({invoke_args})  # raises on failure (launch_kernel throws -> tvm-ffi exception)
'''


class SunmmioSuDeckSourceWrapper:
    """Generate the C++ launcher and the Python dispatch module for one kernel."""

    def __init__(
        self,
        params: list[SunmmioSuDeckParam | tuple[str, str]],
        kernel_name: str,
        elf_name: str,
        launcher_lib_name: str,
    ):
        self.params = [p if isinstance(p, SunmmioSuDeckParam) else SunmmioSuDeckParam(*p) for p in params]
        self.kernel_name = kernel_name
        self.elf_name = elf_name
        self.launcher_lib_name = launcher_lib_name

    def generate_launcher_cpp(self) -> str:
        # Positional arg names: the ABI param names aren't guaranteed valid C identifiers.
        signature_params = ", ".join(
            [
                *(f"{p.c_type} a{i}" for i, p in enumerate(self.params)),
                "int64_t stream_handle",
                "tvm::ffi::Bytes elf",
                "tvm::ffi::Bytes name",
            ]
        )
        pushes = []
        for i, p in enumerate(self.params):
            if p.is_tensor:
                pushes.append(f"  args.emplace_back(*reinterpret_cast<SuTensor *>(static_cast<uintptr_t>(a{i})));")
            else:
                pushes.append(f"  args.emplace_back(a{i});")
        return _LAUNCHER_TEMPLATE.format(signature_params=signature_params, nargs=len(self.params), arg_pushes="\n".join(pushes))

    def generate_python_module(self) -> str:
        names = [f"a{i}" for i in range(len(self.params))]
        call_params = ", ".join([*names, "stream_handle"])
        invoke_args = ", ".join([*names, "stream_handle", "_ELF", "_NAME"])
        return _PY_MODULE_TEMPLATE.format(
            elf_name=self.elf_name,
            kernel_name=self.kernel_name,
            launcher_lib_name=self.launcher_lib_name,
            call_params=call_params,
            invoke_args=invoke_args,
        )
