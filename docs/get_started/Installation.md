# Installation Guide

## Package Names and Coexistence

TileLang-Mesh uses two names:

- Python distribution: `tilelang-mesh`
- Python import package: `tilelang`

Upstream TileLang uses the `tilelang` distribution and the same import package. The two distributions
cannot safely coexist because they install files into the same Python package. Always use a clean
environment, or remove upstream TileLang first:

```bash
python -m pip uninstall -y tilelang
```

To remove TileLang-Mesh later, use:

```bash
python -m pip uninstall tilelang-mesh
```

## Supported Environments

The project metadata requires CPython 3.12 or newer. Published release wheels target Linux x86_64 and
the SunMMIO/SUVM backend. Source builds require CMake 3.26.1 or newer and a C++17 compiler.

Public release artifacts and authorized SUNMMIO source builds have different capabilities:

| Installation path | SUNMMIO/SUVM | Private NPU-IR required |
| --- | --- | --- |
| Public GitHub Release wheel | Yes | No |
| Public frontend-only source build | No | No |
| Authorized recursive source checkout | Yes | Required while building |

The release workflow uses authorized NPU-IR access to build the SunMMIO backend and packages the
three required NPU-IR executables. It explicitly disables CUDA, ROCm, and Metal. The wheel contains
the compiled tools, not the access-controlled NPU-IR source tree, so repository access is not
required at installation time.

## Install a Release Wheel

TileLang-Mesh is distributed through the
[SUNMMIO/Tilelang Releases](https://github.com/SUNMMIO/Tilelang/releases) page. It is not currently
documented as a PyPI package. The command `pip install tilelang` installs upstream TileLang, not this
project.

Create a clean environment, download the correct wheel from the Release assets, and install it:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install /path/to/tilelang_mesh-0.1.0-<platform>.whl
```

Verify the installed distribution outside the source checkout:

```bash
cd /tmp
python -m pip show tilelang-mesh
python -c "from importlib.metadata import version; import tilelang; print(tilelang.__version__); assert tilelang.__version__ == version('tilelang-mesh')"
python -c "from tilelang import tvm; assert tvm.ffi.get_global_func('target.build.tilelang_sunmmio_without_compile', allow_missing=True) is not None"
python -c "from tilelang.jit.adapter.sunmmio.libgen import find_npuir_tool; [find_npuir_tool(name) for name in ('npuir-opt', 'npuir-translate', 'npuir-compile')]"
python -m pip check
```

For release `v0.1.0`, the expected printed version is `0.1.0`.

## Source Archives

This release does not attach a Python source distribution. GitHub's automatic "Source code"
archives omit submodule contents and are not buildable distributions. Clone the repository and use
one of the source-build procedures below instead.

## Validate a Public Checkout

Install Ubuntu/Debian build requirements:

```bash
sudo apt-get update
sudo apt-get install -y \
  git python3 python3-dev python3-setuptools \
  gcc g++ build-essential cmake ninja-build \
  zlib1g-dev libedit-dev libtinfo-dev libxml2-dev
```

If the system CMake is too old:

```bash
python -m pip install --upgrade pip wheel
python -m pip install "cmake>=3.26.1" ninja scikit-build-core cython
```

Clone the canonical repository and initialize only public submodules:

```bash
git clone https://github.com/SUNMMIO/Tilelang.git
cd Tilelang
git submodule update --init --recursive \
  3rdparty/tvm 3rdparty/cutlass 3rdparty/composable_kernel
```

An unauthenticated checkout can build the frontend for import and static validation. It is not a
supported package for executing on SunMMIO hardware:

```bash
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=OFF" \
  python -m pip install . -v
```

## Build With SUNMMIO

SUNMMIO builds require GitHub SSH access to `SUNMMIO/NPU-IR`. A recursive clone should report an
initialized `3rdparty/NPU-IR` submodule before configuration:

```bash
git clone --recursive https://github.com/SUNMMIO/Tilelang.git
cd Tilelang
git submodule update --init --recursive
git submodule status 3rdparty/NPU-IR
```

Build the SunMMIO-only configuration:

```bash
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON" \
  python -m pip install . -v
```

Reuse an existing LLVM source checkout for NPU-IR:

```bash
CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON -DNPUIR_USE_LLVM_SOURCE_DIR=/path/to/llvm-project" \
  python -m pip install . -v
```

The access policy for the NPU-IR repository is separate from its source-code license. Contact the
SUNMMIO maintainers if the submodule checkout is denied.

## Editable Development Install

Use a dedicated environment and select the intended backend explicitly:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel "build[uv]"
python -m pip install -r requirements-dev.txt

# Authorized SUNMMIO development build
CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON" \
  python -m pip install --no-build-isolation --editable . -v
```

For frequent C++ changes, build with CMake and Ninja directly:

```bash
cmake -S . -B build -G Ninja \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_SUNMMIO=ON
ninja -C build
```

After adding new C++ files, rerun CMake before Ninja. To use the source tree directly after building:

```bash
export PYTHONPATH=/path/to/Tilelang:${PYTHONPATH}
python -c "import tilelang; print(tilelang.__version__)"
```

## Build Options

| Option | Purpose |
| --- | --- |
| `USE_SUNMMIO` | Enable the SunMMIO backend and NPU-IR integration |
| `TILELANG_UPDATE_SUBMODULES` | Allow CMake to update Git submodules |
| `NPUIR_USE_LLVM_SOURCE_DIR` | Reuse an existing LLVM source checkout |
| `NO_VERSION_LABEL` | Disable backend and Git local-version suffixes |

The release workflow sets `USE_CUDA=OFF`, `USE_ROCM=OFF`, `USE_METAL=OFF`, and
`USE_SUNMMIO=ON` explicitly.

Release artifacts set `NO_VERSION_LABEL=ON`, so their runtime version exactly matches the Git Tag and
`VERSION` file.

## Troubleshooting

### NPU-IR checkout fails

Obtain authorization for `SUNMMIO/NPU-IR`, or use the frontend-only public checkout described
above. Do not remove the submodule check while leaving `USE_SUNMMIO=ON`.

### Import reports the wrong version

Run:

```bash
python -m pip show tilelang tilelang-mesh
```

If both distributions are installed, create a fresh environment or uninstall both and reinstall only
`tilelang-mesh`.

### Native library cannot be found

Do not import from an unbuilt source checkout. Install the wheel/package, or finish the CMake/Ninja
build before setting `PYTHONPATH`.
