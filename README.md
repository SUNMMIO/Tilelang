<div align="center">

# Tile Language for Distributed Memory

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tile-ai/tilelang) [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord\&logoColor=white)](https://discord.gg/TUrHyJnKPG)

</div>

## The Goal

The goal of this open source project is to extend TileLang (<https://tilelang.com/>) as a unified DSL (domain-specific language) to enable high-performance kernel development for Near-Memory Computing, Distributed Memory AI Accelerators, and Networked Accelerators.

Near-memory computing and distributed memory systems have become key approaches to address the huge computing demand of AI, while networked accelerators further promote the decoupling and coordination of computational resources. To support efficient programming for such emerging heterogeneous computing architectures, we need a unified domain-specific language (DSL) aimed at enabling high-performance kernel development. The design goal is to abstract away underlying hardware differences among different Near-Memory Accelerators or Networked Accelerators, and provide a unified interface, allowing developers to focus on algorithm optimization rather than hardware adaptation. The language likely incorporates key techniques such as tensor tiling, dataflow scheduling, memory layouting, and communication-aware compilation, supporting automatic code generation and optimization to achieve efficient kernel execution across various advanced AI acceleration architectures. Through this unified language framework, we aim to significantly reduce the complexity of cross-platform AI operator development, improving both development efficiency and system performance.

## Installation

TileLang-Mesh is a TileLang variant for distributed-memory and SunMMIO-style backends. It keeps the Python import namespace as `tilelang`, but the Python distribution name in this repository is `tilelang-mesh`.

If you only need upstream TileLang, install it from PyPI with `pip install tilelang` as described in the [upstream TileLang README](https://github.com/tile-ai/tilelang). If you need this distributed-memory variant, install from this repository instead.

### Differences from Upstream TileLang

- Upstream TileLang can be installed directly with `pip install tilelang`; that command does not install TileLang-Mesh.
- This repository builds the `tilelang-mesh` distribution while preserving `import tilelang` compatibility.
- This repository adds a `3rdparty/NPU-IR` submodule for the SunMMIO/SUVM backend. Its SSH URL requires GitHub credentials with access to that repository.
- `USE_SUNMMIO` controls the SunMMIO/SUVM backend. It defaults to `ON`, which integrates NPU-IR and builds the SunMMIO runtime/codegen path. Set `-DUSE_SUNMMIO=OFF` or `USE_SUNMMIO=0` to disable it.
- The usual TileLang backends are still available: CUDA is selected by default on Linux when not explicitly disabled, ROCm can be selected with `USE_ROCM`, and Metal is selected by default on macOS.
- Developer rebuilds use the repository `build` directory. After changing C++ files, rebuild from `build` with `ninja`; after adding new C++ files, rerun CMake first.

### Prerequisites

- Linux is the primary development platform for TileLang-Mesh.
- Python >= 3.9.
- CMake >= 3.26.1 and a C++17 compiler.
- Ninja is recommended for faster native builds.
- CUDA toolkit if building the default CUDA backend.
- Git submodules, including `3rdparty/NPU-IR`, if building with `USE_SUNMMIO=ON`.

On Ubuntu/Debian systems:

```bash
sudo apt-get update
sudo apt-get install -y \
  git python3 python3-dev python3-setuptools \
  gcc g++ build-essential cmake ninja-build \
  zlib1g-dev libedit-dev libtinfo-dev libxml2-dev
```

If your system CMake is too old, install the Python build tools in the target environment:

```bash
python -m pip install -U pip wheel
python -m pip install "cmake>=3.26.1" ninja scikit-build-core cython
```

### Install from Source

For a SunMMIO build, use a recursive clone so the vendored TVM, CUTLASS, Composable Kernel, and NPU-IR sources are present. This requires access to the SSH-based NPU-IR submodule:

```bash
git clone --recursive https://github.com/Sunmmio/Tilelang-mesh.git
cd Tilelang-mesh
git submodule update --init --recursive
```

For a build without SunMMIO and without NPU-IR access, initialize only the other required submodules:

```bash
git clone https://github.com/Sunmmio/Tilelang-mesh.git
cd Tilelang-mesh
git submodule update --init --recursive \
  3rdparty/tvm 3rdparty/cutlass 3rdparty/composable_kernel
```

For that checkout, use the no-SunMMIO install command below; the default build still expects NPU-IR.

Then install in the active Python environment. For normal use, prefer the non-editable install:

```bash
conda activate mesh
python -m pip install . -v
```

If the machine does not have a CUDA toolkit, disable the CUDA backend explicitly. This still builds the default SunMMIO/SUVM backend:

```bash
CMAKE_ARGS="-DUSE_CUDA=OFF" python -m pip install . -v
```

If you do not need the SunMMIO/SUVM backend, disable it explicitly. `TILELANG_UPDATE_SUBMODULES=OFF` prevents CMake from trying to initialize the unavailable NPU-IR submodule:

```bash
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_CUDA=OFF -DUSE_SUNMMIO=OFF" \
  python -m pip install . -v
```

For project development, use an editable install:

```bash
conda activate mesh
python -m pip install -e . -v
```

Verify the import:

```bash
python -m pip show tilelang-mesh
python -c "import tilelang; print('tilelang import OK')"
```

### Common Build Configurations

Default CUDA + SunMMIO/SUVM build, non-editable:

```bash
python -m pip install . -v
```

Default CUDA + SunMMIO/SUVM build for development:

```bash
python -m pip install -e . -v
```

Build without SunMMIO/SUVM after the other required submodules have been initialized:

```bash
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_SUNMMIO=OFF" \
  python -m pip install . -v
```

Build SunMMIO/SUVM with an existing LLVM source checkout for NPU-IR:

```bash
CMAKE_ARGS="-DUSE_SUNMMIO=ON -DNPUIR_USE_LLVM_SOURCE_DIR=/path/to/llvm-project" \
  python -m pip install . -v
```

Build with ROCm and without SunMMIO/SUVM:

```bash
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_CUDA=OFF -DUSE_ROCM=ON -DUSE_SUNMMIO=OFF" \
  python -m pip install . -v
```

### Faster Rebuild for Developers

For frequent C++ development, configure the native build once and rebuild with Ninja:

```bash
mkdir -p build
cd build
cmake .. -G Ninja -DUSE_CUDA=ON -DUSE_SUNMMIO=ON
ninja
```

When using the source tree directly:

```bash
export PYTHONPATH=/path/to/Tilelang-mesh:$PYTHONPATH
python -c "import tilelang; print('tilelang import OK')"
```

After editing existing C++ files:

```bash
cd build
ninja
```

After adding new C++ files:

```bash
cd build
cmake .. -G Ninja -DUSE_CUDA=ON -DUSE_SUNMMIO=ON
ninja
```

### Docker

Dockerfiles are available under `docker/`. The CUDA development image can be built from the repository root:

```bash
docker build -t sunlune/tilelang:cuda -f docker/Dockerfile.cu130.dev .
```

See [docker/README.md](docker/README.md) for container launch examples.

## Documentation

- [SunMMIO TileLang Kernel Quick Start](docs/sunmmio/sunmmio_tilelang_getting_started.md) ([中文](docs/sunmmio/sunmmio_tilelang_getting_started_zh_cn.md)): tutorial covering placement, kernel launch, inter-core communication, layout, and tile-level computation.
- [SunMMIO TileLang User Guide](docs/sunmmio/sunmmio_tilelang_user_guide.md) ([中文](docs/sunmmio/sunmmio_tilelang_user_guide_zh_cn.md)): user-facing guide for writing, migrating, and debugging TileLang kernels on the SunMMIO target.
- [TileLang docs](docs/index.md): full documentation index, including general TileLang guides and SunMMIO-specific notes.
