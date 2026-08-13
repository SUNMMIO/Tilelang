<div align="center">

# TileLang-Mesh

Tile language and compiler extensions for distributed-memory accelerators

</div>

TileLang-Mesh extends [TileLang](https://github.com/tile-ai/tilelang) with abstractions and
compiler support for near-memory computing, distributed-memory AI accelerators, and networked
accelerators. Its release artifacts target the SunMMIO/SUVM backend and include mesh-aware data
placement, communication, and code generation for SunMMIO hardware.

> [!IMPORTANT]
> The Python distribution is named `tilelang-mesh`, but it provides the `tilelang` import package.
> It cannot safely coexist with the upstream `tilelang` distribution in one environment. Uninstall
> upstream TileLang before installing TileLang-Mesh.

## Release Model

The repository has two supported build modes:

| Build | Audience | SUNMMIO backend | NPU-IR access at install time |
| --- | --- | --- | --- |
| Public GitHub Release wheel | SunMMIO users | Enabled | Not required |
| Authorized source build | SunMMIO developers | Enabled | Required while building |

The release workflow uses authorized access to build the SunMMIO backend and package the required
NPU-IR executables. It explicitly disables CUDA, ROCm, and Metal. The wheel does not contain the
access-controlled NPU-IR source tree, so users do not need repository access to install it. The
generated GitHub "Source code" archives omit submodules and are not buildable distributions; use an
attached wheel, or clone the repository for an authorized source build.

## Requirements

- CPython 3.12 or newer
- Linux x86_64 for the published wheel
- CMake 3.26.1 or newer and a C++17 compiler for source builds
- Ninja is recommended
- Access to `SUNMMIO/NPU-IR` for SUNMMIO source builds

The exact Python, operating-system, architecture, and accelerator combinations tested for a release
are listed in that release's compatibility section.

## Install a Public Release

Download the wheel matching your platform from the
[GitHub Releases](https://github.com/SUNMMIO/Tilelang/releases) page, then install it in a clean
environment:

```bash
python -m pip uninstall -y tilelang
python -m pip install /path/to/tilelang_mesh-0.1.0-<platform>.whl
```

Verify the distribution version and packaged SunMMIO components:

```bash
python -m pip show tilelang-mesh
python -c "from importlib.metadata import version; import tilelang; print(tilelang.__version__); assert tilelang.__version__ == version('tilelang-mesh')"
python -c "from tilelang import tvm; assert tvm.ffi.get_global_func('target.build.tilelang_sunmmio_without_compile', allow_missing=True) is not None"
python -c "from tilelang.jit.adapter.sunmmio.libgen import find_npuir_tool; [find_npuir_tool(name) for name in ('npuir-opt', 'npuir-translate', 'npuir-compile')]"
```

TileLang-Mesh is not currently documented as a PyPI install. Do not use `pip install tilelang` for
this project; that command installs upstream TileLang.

## Validate a Public Source Checkout

Clone the canonical repository and initialize only public submodules:

```bash
git clone https://github.com/SUNMMIO/Tilelang.git
cd Tilelang
git submodule update --init --recursive \
  3rdparty/tvm 3rdparty/cutlass 3rdparty/composable_kernel
```

An unauthenticated checkout can build the Python frontend for import and static validation, but the
result is not a supported SunMMIO runtime package:

```bash
python -m pip uninstall -y tilelang
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=OFF" \
  python -m pip install . -v
```

## Build from Source With SUNMMIO

This path requires GitHub access to the SSH-based NPU-IR submodule:

```bash
git clone --recursive https://github.com/SUNMMIO/Tilelang.git
cd Tilelang
git submodule update --init --recursive
python -m pip uninstall -y tilelang
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON" \
  python -m pip install . -v
```

To reuse an existing LLVM source checkout for NPU-IR:

```bash
CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON -DNPUIR_USE_LLVM_SOURCE_DIR=/path/to/llvm-project" \
  python -m pip install . -v
```

## SunMMIO Development

Editable development build:

```bash
CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON" \
  python -m pip install -e . -v
```

For frequent C++ development, configure once and rebuild with Ninja:

```bash
cmake -S . -B build -G Ninja \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON
ninja -C build
```

When running directly from the source tree, add the repository to `PYTHONPATH` only after the native
libraries have been built:

```bash
export PYTHONPATH=/path/to/Tilelang:${PYTHONPATH}
python -c "import tilelang; print(tilelang.__version__)"
```

## Documentation

- [Installation guide](docs/get_started/Installation.md)
- [SunMMIO kernel quick start](docs/sunmmio/sunmmio_tilelang_getting_started.md)
  ([中文](docs/sunmmio/sunmmio_tilelang_getting_started_zh_cn.md))
- [SunMMIO user guide](docs/sunmmio/sunmmio_tilelang_user_guide.md)
  ([中文](docs/sunmmio/sunmmio_tilelang_user_guide_zh_cn.md))
- [Full documentation index](docs/index.md)
- [Contributing guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

See [LICENSE](LICENSE) and [THIRDPARTYNOTICES.txt](THIRDPARTYNOTICES.txt). The access policy for the
NPU-IR repository is separate from the license terms of code included in a release.
