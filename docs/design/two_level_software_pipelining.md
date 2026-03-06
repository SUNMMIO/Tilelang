# Two-Level Software Pipelining Strategy

## 1. Motivation

Modern AI accelerators (e.g., Sunmmio) contain multiple independent hardware units:

- **DMA unit**: Asynchronous data movement between off-chip (DRAM) and on-chip (SRAM).
- **Tensor Core (TC) unit**: Asynchronous matrix multiply-accumulate (MMA).
- **Tile unit**: Synchronous fixed-size 2D-tile arithmetic (e.g., elementwise, reduction).

From the CPU's perspective, DMA and TC are co-processors that run asynchronously —
the CPU issues work to them and continues immediately. The Tile unit is synchronous —
it blocks the CPU until the operation completes.

In a naive schedule, these units operate sequentially, leaving most hardware idle at any
given time. Software pipelining overlaps their execution by reordering operations across
loop iterations. This document describes a **two-level software pipelining strategy**
that systematically exploits all available hardware parallelism.

## 2. Background: Software Pipelining as Stage Skewing

Software pipelining transforms a loop by assigning each operation a **stage number**.
An operation at stage S processes data from iteration `j - S` when the loop counter is
at `j`. This skewing allows operations from different iterations to execute in the same
loop body, enabling overlap when they target independent hardware units.

Given a loop body with operations at stages 0, 1, ..., K, the transformed loop has:

- **Prologue**: K iterations that progressively fill the pipeline.
- **Steady-state body**: Each iteration executes all stages, processing different
  original iterations simultaneously.
- **Epilogue**: K iterations that drain the pipeline.

Intermediate buffers between stages require **multi-versioning**: a buffer written at
stage S_write and read at stage S_read needs `(S_read - S_write + 1)` physical copies,
indexed by `j mod num_versions`, to prevent write-after-read hazards across the skewed
iterations. This count can be reduced by 1 when the execution order within one iteration
guarantees that the read completes before the write to the same version slot.

## 3. Level 1: DMA-Compute Pipeline

### 3.1 The Problem

```
for j in range(N):
    DMA.load(K[j] → K_sram)         # blocks until transfer completes
    DMA.load(V[j] → V_sram)
    T[j] = TC.gemm(Q, K_sram)       # blocks until MMA completes
    C[j] = Tile.softmax(T[j])       # blocks until done
    U[j] = TC.gemm(C[j], V_sram)    # blocks until MMA completes
```

The DMA unit is idle during all compute. The TC and Tile units are idle during all
DMA transfers. Each unit waits for the others.

### 3.2 Stage Assignment

Assign DMA operations to stage 0 and all compute operations to stage `D` (the pipeline
depth), where `D >= 1`:

| Operation            | Stage | Hardware | Skewed Index |
|----------------------|-------|----------|--------------|
| DMA.load(K[j], V[j]) | 0     | DMA      | j            |
| T[j] = Q · K[j]ᵀ    | D     | TC       | j - D        |
| C[j] = softmax(T[j]) | D     | Tile     | j - D        |
| U[j] = C[j] · V[j]  | D     | TC       | j - D        |

### 3.3 Transformed Loop (D = 2, double buffer)

```
# Prologue (D iterations): fill the DMA pipeline
for j in range(0, D):
    DMA.issue(K[j] → K_sram[j % D])
    DMA.issue(V[j] → V_sram[j % D])
    DMA.sync()

# Steady state
for j in range(D, N):
    DMA.issue(K[j] → K_sram[j % D])       # stage 0: prefetch
    DMA.issue(V[j] → V_sram[j % D])
    DMA.sync()
    T    = TC.gemm(Q, K_sram[(j-D) % D])  # stage D: compute
    C    = Tile.softmax(T)
    U    = TC.gemm(C, V_sram[(j-D) % D])

# Epilogue (D iterations): drain remaining compute
for j in range(N, N + D):
    T    = TC.gemm(Q, K_sram[(j-D) % D])
    C    = Tile.softmax(T)
    U    = TC.gemm(C, V_sram[(j-D) % D])
```

### 3.4 What This Achieves

```
DMA:     ██load[2]██  ██load[3]██  ██load[4]██
Compute: ██ T[0] C[0] U[0] ██  ██ T[1] C[1] U[1] ██  ██ T[2] ...
         ├──── overlap ────┤  ├──── overlap ────┤
```

DMA latency is hidden behind compute. The depth D controls how far ahead DMA
prefetches. Larger D tolerates more DMA latency variation at the cost of D × tile_size
bytes of SRAM per operand.

### 3.5 What Remains Sequential

All compute operations within one iteration still execute sequentially:
`T → C → U`. The Tensor Core is idle during the entire softmax phase, and the Tile
unit is idle during both GEMMs. Level 2 addresses this.

## 4. Level 2: Intra-Compute Pipeline (TC-Tile Overlap)

### 4.1 The Problem

Within the compute phase of each iteration:

```
T[j] = TC.gemm(Q, K[j]ᵀ)       # TC async
C[j] = Tile.softmax(T[j])       # Tile sync (CPU blocks here)
U[j] = TC.gemm(C[j], V[j])      # TC async
```

Timeline:

```
TC:   ██ T[j] ██              ██ U[j] ██
Tile:             ██ C[j] ██
                  ↑ TC idle    ↑ Tile idle
```

### 4.2 Dependency Analysis

The operation dependencies are:

```
Within iteration j:    T[j] → C[j] → U[j]
Across iterations:     U[j-1] → U[j]  (accumulation)

T[j] is independent of C[j-1], U[j-1], and T[j-1].
T[j] only requires K[j] (from DMA) and Q (constant/reused).
```

The pair **(T[j], C[j-1])** satisfies both conditions for overlap:
1. **Data-independent**: T[j] needs K[j] and Q; C[j-1] needs T[j-1]'s output. No
   shared dependency.
2. **Different hardware units**: T[j] runs on TC (async), C[j-1] runs on Tile (sync).

No other cross-iteration pair satisfies both conditions:
- U[j-1] and C[j-1]: dependent (U[j-1] needs C[j-1]'s output).
- T[j] and U[j-1]: independent but same hardware unit (both TC).
- U[j-1] and C[j]: C[j] needs T[j], which hasn't executed yet.

### 4.3 Stage Assignment

Assign T to sub-stage 0 and {C, U} to sub-stage `P` (the compute pipeline depth),
where `P >= 1`:

| Operation            | Sub-stage | Hardware | Skewed Index |
|----------------------|-----------|----------|--------------|
| T[j] = Q · K[j]ᵀ    | 0         | TC       | j            |
| C[j] = softmax(T[j]) | P         | Tile     | j - P        |
| U[j] = C[j] · V[j]  | P         | TC       | j - P        |

### 4.4 Transformed Compute Phase (P = 1)

```
# Prologue: bootstrap the compute pipeline
TC.issue(T[0])
TC.sync()                       # unavoidable: C[0] needs T[0]

# Steady state, j = 1, 2, ..., N-1:
for j in range(1, N):
    TC.issue(T[j])              # (1) TC starts T[j], returns immediately
    C[j-1] = Tile.softmax(...)  # (2) Tile blocks CPU; TC computes T[j] in background
    TC.sync()                   # (3) wait for T[j] to complete
    TC.issue(U[j-1])            # (4) TC starts U[j-1], returns immediately
    TC.sync()                   # (5) wait for U[j-1] to complete

# Epilogue
C[N-1] = Tile.softmax(...)
TC.issue(U[N-1])
TC.sync()
```

Step-by-step correctness:

| Step | Operation         | Requires              | Satisfied Because                        |
|------|-------------------|-----------------------|------------------------------------------|
| (1)  | TC.issue(T[j])    | K[j] in SRAM, Q ready | DMA loaded K[j] (level 1); Q is constant |
| (2)  | Tile.softmax(j-1) | T[j-1] complete       | TC.sync() in previous iteration's step 3  |
| (3)  | TC.sync()         | T[j] complete         | Blocks until TC finishes                  |
| (4)  | TC.issue(U[j-1])  | C[j-1] complete       | Step 2 is sync — already finished         |
| (5)  | TC.sync()         | U[j-1] complete       | Blocks until TC finishes                  |

### 4.5 Resulting Timeline

```
Overlapped:
TC:   ▓T[0]▓ ██T[1]██  ██U[0]██  ██T[2]██  ██U[1]██  ██T[3]██  ██U[2]██
Tile:         ██C[0]██             ██C[1]██             ██C[2]██
              ├overlap┤            ├overlap┤            ├overlap┤
              T[1]∥C[0]           T[2]∥C[1]           T[3]∥C[2]

vs. Naive:
TC:   ▓T[0]▓          ██U[0]██  ▓T[1]▓          ██U[1]██  ▓T[2]▓
Tile:         ██C[0]██                   ██C[1]██                   ██C[2]██
              ↑ TC idle                  ↑ TC idle                  ↑ TC idle
```

Savings per iteration: `min(time(T), time(C))`. When T and C take similar time, this
saves approximately one GEMM's worth of latency per iteration.

### 4.6 When Additional Tile Work Exists

If there are more synchronous Tile operations (e.g., output rescaling in online
softmax), they can fill the remaining gap at step (4)-(5):

```
    TC.issue(U[j-1])            # (4) TC starts U[j-1]
    Tile.rescale_O(j-2)         # (4') Tile work overlaps with U[j-1] on TC
    TC.sync()                   # (5) wait for U[j-1]
```

The general principle: **every synchronous Tile operation is a window to hide TC
latency. The schedule should pair each Tile operation with a preceding TC.issue().**

## 5. Composing Both Levels

### 5.1 Flattened Stage Assignment

When both levels are applied, the stages compose additively. With DMA depth D and
compute depth P:

| Operation            | Flat Stage | Skewed Index | Hardware | Buffer Space |
|----------------------|------------|--------------|----------|--------------|
| DMA.load(K[j], V[j]) | 0          | j            | DMA      | —            |
| T[j] = Q · K[j]ᵀ    | D          | j - D        | TC       | SRAM → RSRAM |
| C[j] = softmax(T[j]) | D + P      | j - D - P    | Tile     | RSRAM        |
| U[j] = C[j] · V[j]  | D + P      | j - D - P    | TC       | RSRAM        |

Total prologue length: D + P iterations.

### 5.2 Independent Buffer Versioning

Each level versions buffers in a **different memory space**:

```
Level 1 buffers (SRAM):
    K_sram: D versions, indexed by j mod D
    V_sram: D versions, indexed by j mod D
    Cost: D × tile_size × num_operands in SRAM

Level 2 buffers (RSRAM):
    S (T→C intermediate): P versions, indexed by j mod P
    Cost: P × tile_size in RSRAM
```

This is why two separate transformations are preferable to one flat pipeline —
a flat `num_stages = D + P` would over-provision buffer copies (allocating D + P
versions in SRAM when only D are needed, and D + P in RSRAM when only P are needed).

### 5.3 Complete Pseudo-Code (D = 2, P = 1)

```
# ============================================================
# Level 1 Prologue: fill DMA pipeline (D = 2 iterations)
# ============================================================
DMA.issue(K[0] → K_sram[0]);  DMA.issue(V[0] → V_sram[0]);  DMA.sync()
DMA.issue(K[1] → K_sram[1]);  DMA.issue(V[1] → V_sram[1]);  DMA.sync()

# ============================================================
# Level 2 Prologue: fill TC-Tile pipeline (P = 1 iteration)
# Process first data point (j_compute = 0)
# ============================================================
TC.issue(T[0], Q, K_sram[0 % 2])
TC.sync()

# ============================================================
# Steady state: j = D + P ... N - 1  (all units active)
# ============================================================
for j in range(D + P, N):
    # --- Level 1: DMA prefetch (stage 0) ---
    DMA.issue(K[j] → K_sram[j % D])
    DMA.issue(V[j] → V_sram[j % D])

    # --- Level 2: overlapped TC + Tile (stages D and D+P) ---
    j_T = j - D           # iteration index for T
    j_C = j - D - P       # iteration index for C and U

    TC.issue(T[j_T])               # TC starts T, returns immediately
    C[j_C] = Tile.softmax(...)     # Tile blocks CPU; TC runs in background
    TC.sync()                      # T[j_T] complete
    TC.issue(U[j_C])               # TC starts U, returns immediately
    TC.sync()                      # U[j_C] complete

    DMA.sync()                     # ensure DMA for this iteration done

# ============================================================
# Epilogue: drain remaining compute
# ============================================================
# (D + P iterations of compute without new DMA)
```

### 5.4 Steady-State Hardware Utilization

```
DMA:  ══load[j]══════════════════════════════  (async, overlapped with everything)
TC:   ██ T[j-D] ██  ██ U[j-D-P] ██            (two async operations per iteration)
Tile:  ██ C[j-D-P] ██                          (sync, overlaps with T on TC)
      ├── all three units active ──────────┤
```

## 6. Relationship to Existing Frameworks

### 6.1 Tilelang's InjectSoftwarePipeline

Tilelang's current software pipelining pass (`inject_pipeline.cc`) implements Level 1.
It assigns `(stage, order)` annotations to each statement, computes buffer versions via
def-use analysis (`ComputeBufferVersions`), and transforms the loop into
prologue/body/epilogue with multi-versioned buffers (`MultiVersionBuffer`).

The same machinery can express Level 2 by assigning **finer stage numbers** to
operations within the compute phase. The key extension needed is a pipeline planner that
recognizes hardware unit boundaries (not just DMA-vs-compute) as stage boundaries.

### 6.2 Tawa's Aref Abstraction

The Tawa compiler (arXiv 2510.14719) unifies both levels under a single IR abstraction
called **asynchronous references (aref)**. Each aref is a typed channel with `put`,
`get`, and `consumed` operations backed by hardware barriers (mbarriers on NVIDIA
Hopper GPUs).

- **Aref depth D** corresponds to Level 1 (DMA-compute buffering).
- **MMA pipeline depth P** corresponds to Level 2 (TC-Tile overlap within consumer).

Both parameters control how long a consumer holds buffer slots before releasing them,
expressed uniformly through the aref lifecycle.

### 6.3 Correspondence Table

| Concept                     | This Document  | Tilelang              | Tawa              |
|-----------------------------|----------------|-----------------------|--------------------|
| DMA-Compute buffer depth    | D              | num_stages            | aref depth D       |
| TC-Tile overlap depth       | P              | (not yet supported)   | MMA depth P        |
| DMA buffer versions         | D              | ComputeBufferVersions | aref slot count    |
| TC-Tile buffer versions     | P              | (not yet supported)   | implicit in P      |
| DMA buffer memory           | SRAM           | shared memory         | shared memory      |
| TC-Tile buffer memory       | RSRAM          | (N/A)                 | registers          |
| Stage assignment            | flat stages    | (stage, order)        | producer/consumer  |
| Loop transformation         | skew + unroll  | prologue/body/epilogue| warp specialization|

## 7. Design Considerations

### 7.1 Choosing D (DMA Depth)

- **D = 2** (double buffer): Sufficient when one iteration of compute takes longer than
  one DMA transfer. Most common choice.
- **D = 3** (triple buffer): Useful when DMA latency is variable or compute per
  iteration is short. Adds one extra SRAM copy per operand.
- **D > 3**: Diminishing returns. DMA latency is usually bounded and 3 buffers are
  enough to absorb variance.

### 7.2 Choosing P (Compute Pipeline Depth)

- **P = 1**: The primary useful setting. Overlaps T[j] with C[j-1] using 1 extra RSRAM
  copy. Saves `min(time(T), time(C))` per iteration.
- **P = 2**: Would require 2 extra RSRAM copies and a deeper prologue. Only beneficial
  if `time(C) > time(T)` — i.e., the Tile operation is so long that one TC operation
  isn't enough to fill the gap. Rare in practice.
- **P > 2**: Unlikely to help. RSRAM is scarce and the overlap opportunity is bounded
  by the single Tile-TC pair.

### 7.3 Resource Constraints

The total on-chip memory budget constrains both D and P simultaneously:

```
SRAM usage  = D × (size(K_tile) + size(V_tile))
RSRAM usage = P × size(S_tile) + size(O_accumulator)
```

Since SRAM and RSRAM are independent memory spaces, D and P can be chosen
independently — increasing D does not reduce the budget available for P, and vice versa.

### 7.4 Applicability Beyond FlashAttention

This two-level strategy applies to any loop body with the pattern:

```
for j in range(N):
    DMA(j)                     # async unit A
    async_compute_1(j)         # async unit B
    sync_compute(j)            # sync (blocks CPU)
    async_compute_2(j)         # async unit B
```

Where `async_compute_1[j]` is independent of `sync_compute[j-1]`. Examples include:
- **GEMM with epilogue fusion**: MMA (TC) → elementwise activation (Tile) → store.
- **Convolution**: MMA (TC) → bias add + activation (Tile).
- **Layer norm / RMS norm after GEMM**: MMA (TC) → reduction + scale (Tile).
