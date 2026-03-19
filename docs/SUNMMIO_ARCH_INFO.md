# SunMMIO NPU Architecture Info (for Trae Handoff)

## Target Identity

- Canonical target string used in this repo:
  - `llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_4,device_mesh_ncol_4`
- The user-facing alias `"Sunmmio"` maps to the target string above.
- SunMMIO is detected by TileLang as:
  - target device type is CPU (`kDLCPU`)
  - `mcpu` starts with `sunmmio-`

## Topology and Memory Model

- Current mesh config is modeled as **4 x 4** cores (`nrow=4`, `ncol=4`), total **16** cores.
- Static per-core SRAM model in driver:
  - `RSRAM`: `1,536,000` bytes
  - `WSRAM`: `2,097,152` bytes
  - `ASRAM`: `1,048,576` bytes
- Mesh dimensions are passed through target attributes:
  - `mattr=device_mesh_nrow_4,device_mesh_ncol_4`

## Programming Model Assumptions

- SunMMIO pipeline is treated as **threadless** in current transform/tests:
  - no `threadIdx.*`
  - no `v_thread`
  - `blockIdx.*` preserved for grid semantics
- Communication ops (`broadcast`, `put`, `allgather`, `allreduce`) are SunMMIO-only and validate:
  - target is SunMMIO
  - core id bounds against mesh shape
  - size/shape constraints

## Current Codegen Status

- `target.build.tilelang_sunmmio`:
  - intentionally **not implemented** right now (fails with clear message)
- `target.build.tilelang_sunmmio_without_compile`:
  - implemented and returns a **source module** (`type_key: sunmmio`)
  - emits pseudo-MLIR-like text from `codegen_sunmmio.cc`

## Pseudo-MLIR Dialect Snapshot

- Statements: `sunmmio.module`, `sunmmio.func`, `sunmmio.for`, `sunmmio.if`, `sunmmio.return`
- Memory ops: `sunmmio.alloc`, `sunmmio.load`, `sunmmio.store`
- Arithmetic ops: `sunmmio.add`, `sunmmio.sub`, `sunmmio.mul`, `sunmmio.div`
- Other ops: `sunmmio.cast`, `sunmmio.call`, `sunmmio.eval`

## Quick Verification Commands (serve)

- Run SunMMIO skeleton codegen test (prints pseudo-MLIR):
  - `/opt/miniconda3/bin/conda run -n tl python -m pytest -q -s testing/python/target/test_sunmmio_codegen_skeleton.py`
- Optional build sanity:
  - `/opt/miniconda3/bin/conda run -n tl ~/.local/bin/cmake -S . -B build -DTILELANG_UPDATE_SUBMODULES=OFF`
  - `/opt/miniconda3/bin/conda run -n tl ~/.local/bin/cmake --build build -j8`

## Key Source References

- Target mapping and alias:
  - `tilelang/utils/target.py`
- Device properties / mesh model:
  - `tilelang/carver/arch/driver/sunmmio_driver.py`
- Target detection helpers:
  - `src/target/utils.cc`
- Source-only runtime build path:
  - `src/target/rt_mod_sunmmio.cc`
- Pseudo-MLIR codegen:
  - `src/target/codegen_sunmmio.cc`
- SunMMIO communication semantics:
  - `src/op/comm.cc`
