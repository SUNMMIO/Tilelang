# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TileLang is a unified DSL (domain-specific language) for high-performance kernel development targeting Near-Memory Computing, Distributed Memory AI Accelerators, and Networked Accelerators. It extends the open-source Tilelang project to support chips with distributed memory architectures.

**Key Concepts:**
- **Tensor Tiling**: Breaking down tensors into manageable tile units for processing
- **Memory Layouts**: Sophisticated layout system supporting fragments, swizzling, and hierarchical layouts
- **MeshTensor**: Distributed tensor abstraction for multi-chip execution
- **Compilation Pipeline**: Python DSL → TVM IR → Target-specific code (CUDA/ROCm/Metal)

### Sunmmio Target

We are in particular working on Tilelang for the Sunmmio Target.
1. Register Sunmmio Target in Tilelang
2. Implement custom Ops for Target
3. Layout Inference & Special lowering for Sunmmio Target
4. Target-specific optimizations for Sunmmio Target, mainly focusing on Software pipelining
5. CodeGen for Sunmmio Target

### Characteristics of Sunmmio Target

#### NPU (SIMD)

The biggest difference between Sunmmio target and Nvidia target is that Sunmmio target is SIMD-based rather than SIMT.
That being said, we only have one single thread in each core. Hardware cannot help software perform any scheduling.
Hence, software pipelining is extremely important.

#### Basic Unit in Target

There are three main hardware units in each core of Sunmmio target, namely,
1. DMA unit for data movement between on-chip storage and off-chip storage,
2. Tensor Core unit for matrix multiplication and accumulaion.
3. Tile unit for fixed size 2D-tile arithmetics, e.g., addition of two tiles of shape (16 x 32 x bf16), reduction on one axes of a tile of shape (8 x 32 x fp32).
4. Scalar unit for simple scalar arithemtics and control.

From the perspective of CPU (the main thread in each code), DMA unit and Tensor Core are co-processors that run asynchronously.
We have explicit sycn instructions to wait for the finish of these units.

#### Memory Hierarchy

There are three on-chip storages in a core, namely, ASRAM, WSRAM and RSRAM.
ASRAM and WSRAM are special buffers for Tensor unit. They are used to store the left operand and right operand of an MMA.
The MMA results will be streamed to RSRAM, which can then post-processed by the Tile unit.

Data in DRAM can be copied directly to ASRAM, WSRAM and RSRAM.
Data in RSRAM can also be copied to ASRAM and WSRAM.

#### Distributed Memory

The final characteristics of Sunmmio target is that cores are organized as a mesh structure, typically 4 cores per row and 4 cores per column.
Each core has its own DRAM, and it cannot access the memory of other cores directly.
To achieve so, it has to leverage two special links on the chip.
1. HLink. This link is responsible for gathering data from each core in a row. We can think of it as a collective all-gather operations. Each core in a row broadcast its data in either DRAM, ASRAM or RSRAM to the others.
2. VLink. Similar to HLink, it connects cores in a column. Each core in a column can broadcast its data in either DRAM, WSRAM and RSRAM to the other cores.

## Architecture

### High-Level Components

1. **Python DSL Layer** (`tilelang/language/`)
   - `kernel.py`: Kernel launch abstraction and thread/block binding
   - `allocate.py`: Memory allocation primitives (shared, local, fragment, barrier, tmem)
   - `gemm.py`, `copy.py`, `reduce.py`, `fill.py`: High-level tile operations
   - `loop.py`: Loop annotations (Parallel, Persistent, Pipelined, serial, unroll)
   - `atomic.py`: Atomic operations for synchronization
   - `v2/`: Version 2 API with `prim_func` decorator and improved type annotations
   - `ast/`: AST nodes for parsing Python code into TVM IR

2. **Layout System** (`tilelang/layout/`, `src/layout/`)
   - `Layout`: Base layout abstraction for memory access patterns
   - `Fragment`: Register-level layout for tensor cores
   - `HierarchicalLayout`: Multi-level layout for distributed memory
   - Swizzle patterns: Volta, WGMMA, TCGEN05, bank conflict avoidance
   - C++ implementation handles layout inference and transformations

3. **JIT Compilation** (`tilelang/jit/`)
   - `compile()`: Main entry point for compiling PrimFunc to executable kernel
   - `jit()`, `lazy_jit()`: Decorators for automatic compilation
   - `JITKernel`: Wrapper around compiled kernel with execution backends
   - Execution backends: tvm_ffi, dlpack, ctypes, cython, nvrtc, torch
   - `par_compile()`: Parallel compilation for multiple kernels

4. **Compiler Passes** (`tilelang/transform/`, `src/transform/`)
   - `LayoutInference()`: Infer fragment/shared memory layouts from operations
   - `ClusterPlanning()`, `PipelinePlanning()`: Multi-stage optimization
   - `WarpSpecializedPipeline()`: Warp-level task specialization
   - `InjectSoftwarePipeline()`: Software pipelining for overlap compute/memory
   - `LowerTileOp()`: Lower high-level tile ops to TVM intrinsics
   - `InjectPTXAsyncCopy()`: CUDA async copy injection
   - `MergeSharedMemoryAllocations()`: Shared memory reuse optimization
   - Many CUDA-specific passes: TMA barriers, WGMMA sync, Hopper intrinsics

5. **Engine** (`tilelang/engine/`)
   - `lower()`: Main lowering function from DSL to executable
   - `KernelParam`: Kernel parameter handling
   - `register_cuda_postproc()`, `register_hip_postproc()`: Backend-specific postprocessing hooks

6. **C++ Backend** (`src/`)
   - `ir.cc`: Core IR nodes and registration
   - `target/`: Code generation for CUDA (`codegen_cuda.cc`), HIP (`codegen_hip.cc`), Metal (`rt_mod_metal.cc`)
   - `runtime/`: Runtime support (error handling, CUDA runtime wrapper)
   - `op/`: Tile operation implementations
   - Built on top of TVM as a submodule (`3rdparty/tvm`)

7. **TileOp Library** (`tilelang/tileop/`)
   - `GemmPy`, `GemmSPPy`: High-level Python implementations of GEMM operations
   - Can be used as building blocks for complex kernels

### Compilation Flow

1. User writes kernel using `@tl.prim_func` decorator or `T.prim_func()` API
2. Python DSL code is parsed into TVM TIR (Tensor Intermediate Representation)
3. Series of transformation passes optimize and lower the IR
4. Backend codegen produces CUDA/HIP/Metal code
5. Runtime compilation (via TVM or NVRTC) produces executable binary
6. JITKernel wrapper provides Python interface for execution

### Key Dependencies

- **TVM**: Core compiler infrastructure (vendored in `3rdparty/tvm`)
- **CUTLASS**: NVIDIA CUDA Templates (`3rdparty/cutlass`)
- **Composable Kernel**: AMD ROCm kernels (`3rdparty/composable_kernel`)
- **PyTorch**: Tensor interface and Metal backend execution
- **apache-tvm-ffi**: Python bindings for TVM runtime

## Development Commands

### Setup

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:<your username>/tilelang.git
cd tilelang

# Setup virtual environment
uv venv --seed .venv  # or: python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python3 -m pip install --upgrade pip setuptools wheel "build[uv]"
uv pip install --requirements requirements-dev.txt

# Install pre-commit hooks
pre-commit install --install-hooks

# Install in editable mode
python3 -m pip install --no-build-isolation --verbose --editable .
```

### Build and Test

```bash
# Run all tests
python3 -m pytest testing

# Run specific test directory
python3 -m pytest testing/python/jit
python3 -m pytest testing/python/language

# Run single test file
python3 -m pytest testing/python/jit/test_tilelang_jit_gemm.py

# Run with verbose output
python3 -m pytest testing/python/jit -v

# Run specific test function
python3 -m pytest testing/python/jit/test_tilelang_jit_gemm.py::test_function_name
```

### Faster C++ Rebuild for Developers

After initial `pip install -e . -v --no-build-isolation`, use ninja for fast C++ rebuilds:

```bash
# Rebuild C++ changes (from build directory)
cd build && ninja

# Or rebuild specific target
cd build && ninja tilelang
```

Alternatively, build via PYTHONPATH without pip:

```bash
# Configure and build
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON   # or -DUSE_CUDA=OFF for CPU-only
ninja -j$(nproc)

# Set PYTHONPATH to use the build
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
python -c "import tilelang; print(tilelang.__version__)"
```

Useful CMake options:
- `-DUSE_CUDA=ON|OFF` - NVIDIA CUDA support (default ON when CUDA found)
- `-DUSE_ROCM=ON` - AMD ROCm support
- `-DUSE_METAL=ON` - Apple Metal support (default ON on macOS)

### Linting and Formatting

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Format changed files only
bash format.sh

# Format all files
bash format.sh --all

# The project uses:
# - yapf for Python formatting (configured in pyproject.toml)
# - clang-format for C++ formatting (.clang-format)
# - ruff for Python linting
# - codespell for spell checking
```

### Backend Selection

The build system auto-detects backends but can be overridden:

```bash
# Force CUDA backend
USE_CUDA=ON python3 -m pip install --no-build-isolation --verbose --editable .

# Force ROCm backend
USE_ROCM=/opt/rocm python3 -m pip install --no-build-isolation --verbose --editable .

# Force Metal backend (macOS)
USE_METAL=ON python3 -m pip install --no-build-isolation --verbose --editable .

# CPU-only build
USE_CUDA=OFF python3 -m pip install --no-build-isolation --verbose --editable .
```

## Code Patterns

### Writing a Kernel

```python
import tilelang.language as T

@T.prim_func
def my_kernel(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float32"),
):
    # Kernel implementation using TileLang DSL
    with T.Kernel(grid, block) as (bx, by):
        # Thread/block bindings available as bx, by
        # Use T.alloc_shared, T.alloc_fragment for memory
        # Use T.copy, T.gemm, T.fill for operations
        pass

# Compile and execute
kernel = tl.compile(my_kernel, target="cuda")
kernel(a_tensor, b_tensor, c_tensor)
```

### Memory Allocation

- `T.alloc_shared()`: Shared memory (GPU thread block)
- `T.alloc_local()`: Register/local memory
- `T.alloc_fragment()`: Register fragments for tensor cores
- `T.alloc_barrier()`: CUDA barriers for synchronization
- `T.alloc_tmem()`: Tensor memory (special memory hierarchy)
- `T.empty()`: Create output tensors (automatically handled)

### Layout Specification

Layouts control how tensors map to memory and hardware:

```python
from tilelang.layout import make_swizzled_layout, Layout

# Create swizzled layout for shared memory
layout = make_swizzled_layout((M, K), (16, 16))

# Use in allocation
shared_mem = T.alloc_shared((M, K), dtype, layout=layout)
```

### Compiler Pass Configuration

```python
# Configure passes when compiling
kernel = tl.compile(
    my_kernel,
    target="cuda",
    pass_configs={
        tl.PassConfigKey.ENABLE_SOFTWARE_PIPELINE: True,
        tl.PassConfigKey.LAYOUT_INFERENCE_VERBOSE: False,
    },
)
```

## Testing

- Tests are in `testing/python/` organized by component
- Use pytest fixtures from `conftest.py` for test configuration
- Example-based tests are in `examples/` subdirectories
- Each example typically has a `test_*.py` file
- Test naming: `test_tilelang_<component>_<feature>.py`

## Build System

- Uses `scikit-build-core` with CMake backend
- `CMakeLists.txt`: Main build configuration
- `pyproject.toml`: Python package metadata and build settings
- Backend detection: Auto-selects CUDA (default on Linux), Metal (default on macOS), or ROCm
- ccache/sccache automatically used if available
- Cython extension for fast FFI adapter generated during build

## Common Issues

1. **Submodule not initialized**: Run `git submodule update --init --recursive`
2. **CUDA version mismatch**: Ensure CUDA toolkit matches build requirements
3. **Build cache issues**: Clear `build/` directory and rebuild
4. **TVM FFI version conflicts**: Check `requirements.txt` for pinned versions

## Branch Conventions

- Main development branch: `tilelang_mesh_main`
- Feature branches: `u/<username>/<feature-name>` (e.g., `u/jiaqiguo/tilevew`)
- Create PRs against `tilelang_mesh_main`
