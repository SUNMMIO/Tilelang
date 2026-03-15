# Pipelined Buffer Dependency Analysis

This note explains TileLang's pipelined buffer dependency analysis, implemented by `tl.transform.AnalyzeBufferDependency`.

The pass analyzes a serial `T.Pipelined` loop and classifies the memory values produced inside that loop into:

- **loop-carried state**: values that are read by a later iteration
- **intra-iteration channels**: values whose `RAW` uses are satisfied within the same iteration

The analysis is attached to the loop as a typed annotation and is intended to feed later Sunmmio pipeline planning and buffer versioning passes.

## Goal of the Analysis

Software-pipelined kernels need a precise answer to three questions:

1. Which buffers carry semantic state across loop iterations?
2. Which buffers are only temporary channels within one iteration?
3. Which structural patterns require special handling or conservative fallback?

This distinction is important for Sunmmio because pipeline planning and multi-versioning are only correct when they preserve true inter-iteration dependences and do not confuse them with plain storage reuse.

The pass is not trying to solve full general TIR memory dependence. Its job is narrower:

- analyze one pipelined serial loop
- recover `RAW` dependences with distance `0` and `1`
- summarize the result in a form that downstream transforms can consume directly

## What The Pass Produces

The canonical result is a typed loop annotation:

- `tl_buffer_dependency_analysis`

The object is defined in `src/analysis/pipelined_buffer_dependency.h` and contains:

- `BufferDependencyInfo`
  - `buffer`
  - `state_regions`
  - `channel_regions`
- `BufferDependencyEdge`
  - `dep_kind`
  - `distance`
  - `buffer`
  - `src_region`
  - `dst_region`
  - `src_effect_id`
  - `dst_effect_id`
- `BufferDependencyPattern`
  - `kind`
  - `buffer`
  - `regions`
  - `effect_ids`
  - `detail`

The current dependence kind is:

- `RAW`

The current structural patterns are:

- `covered_rewrite`
- `mixed_role_regions`
- `partial_overwrite_remainder_read`
- `unknown_effect`

The pass also emits derived debug annotations for inspection and tests:

- `tl_buffer_dependency_state_buffers`
- `tl_buffer_dependency_channel_buffers`
- `tl_buffer_dependency_intra_raw`
- `tl_buffer_dependency_inter_raw`
- `tl_buffer_dependency_mixed_role_buffers`
- `tl_buffer_dependency_mixed_role_details`
- `tl_buffer_dependency_partial_overwrite_hazards`
- `tl_buffer_dependency_covered_rewrites`
- `tl_buffer_dependency_unknown_effects`

These string annotations are views over the typed object. Downstream passes should consume the typed object, not re-parse the strings.

## How The Analysis Works

Before looking at the concrete traversal, it helps to fix the mental model.

This pass does **not** build ordinary scalar SSA. Instead, it treats the pipelined loop as a **region-sensitive memory dataflow problem**:

- the loop body is reduced to an ordered stream of memory effects
- each effect reads or writes one or more `BufferRegion`s
- the loop backedge carries the memory definitions that survive to loop exit
- the current iteration is analyzed with those carried definitions already in scope

So the problem is modeled as:

1. recover the ordered memory effects of one logical iteration
2. compute which memory definitions survive to loop exit
3. feed those surviving definitions through the loop backedge
4. replay one iteration to recover `RAW` edges with distance `0` and `1`
5. classify produced regions into `state`, `channel`, and structural patterns

Conceptually:

```text
one logical iteration
    |
    v
ordered memory effects over BufferRegions
    |
    v
loop-exit reaching definitions
    |
    | shift producer view by k -> k - 1
    v
loop-header carried definitions
    |
    v
forward replay of current iteration
    |
    +--> distance-0 RAW edges  -> intra-iteration channels
    +--> distance-1 RAW edges  -> loop-carried state
    +--> write/carried interactions -> structural patterns
```

Another useful way to think about it is:

- **intra-iteration** means a read is satisfied by a definition created earlier in the same iteration
- **loop-carried** means a read is satisfied by a definition that survived the previous iteration and re-entered at the loop header

This is why the analysis first computes loop-exit definitions and only then analyzes the current iteration. Without that explicit backedge model, it is easy to confuse “written in the previous iteration” with “actually visible in the current iteration before being overwritten”.

### High-Level Workflow

At a high level, the pass solves the problem in three stages:

- **Normalize one iteration into ordered memory effects**
  - Recursively walk the nested TIR, preserve execution order, and summarize leaf accesses as relaxed `BufferRegion`s.
- **Construct loop-carried memory state**
  - Compute which definitions survive to loop exit, then reinterpret them as incoming carried definitions at the next loop header.
- **Resolve dependences and classify**
  - Replay the current iteration from those header definitions, emit `RAW` edges, and derive state/channel/pattern results.

The detailed steps below are just the concrete implementation of that model.

### 1. Select The Loop

The pass looks for serial `For` loops with a `num_stages` annotation. Those are treated as the pipelined loops that need dependency classification.

### 2. Walk Nested TIR In Execution Order

The recursive walker in `src/analysis/analyze_pipelined_buffer_dependency.cc` traverses the loop body and handles common TIR control-flow and scoping nodes explicitly:

- `SeqStmt`
- `BlockRealize`
- `Block`
- `BufferRealize`
- `AttrStmt`
- `AssertStmt`
- `LetStmt`
- `Allocate`
- `AllocateConst`
- `DeclBuffer`
- `For`
- `While`
- `IfThenElse`

This is important because the analysis must respect sequencing inside nested blocks and loops. It is not enough to summarize only the immediate children of the pipelined loop.

### 3. Extract Ordered Memory Effects

At the leaves, the pass collects read/write regions from:

- `BufferStore`
- `BufferLoad`
- `dma_copy`
- `mma_sunmmio`
- `tvm_access_ptr`

The collector also tracks active inner-loop domains and `let` bindings. When it sees a leaf access like `acc[i]`, it relaxes that symbolic access through the current loop domains using `EvalSet` and `CoverRange`.

That means a nested loop such as:

```python
for i in T.serial(4):
    acc[i] = acc[i] * scale[0]
```

is summarized as a write to `acc[0:4]`, not a single point access to `acc[i]`.

### 4. Build Loop-Exit Reaching Definitions

The pass first analyzes one iteration of the loop to compute the memory definitions that survive to loop exit.

Only surviving definitions matter for loop-carried state. If a region is overwritten later in the same iteration, it is not exported through the backedge.

### 5. Seed The Loop Header

The surviving loop-exit definitions are shifted with `k -> k - 1` and reintroduced at the loop header as distance-1 reaching definitions.

Conceptually, this is a lightweight memory-phi for the loop backedge.

### 6. Run Forward Dependence Resolution

The pass then walks the current iteration with those loop-header defs in scope:

- reads consult the current reaching-def set
- writes kill overlapping defs and create new local defs

This produces:

- `distance = 0` edges for same-iteration `RAW`
- `distance = 1` edges for loop-carried `RAW`

The state/channel split is derived from those `RAW` edges:

- a written region that participates in a distance-1 `RAW` is state
- a written region with only distance-0 `RAW` is a channel

### 7. Emit Structural Patterns

The pass also emits structural facts that are useful to later planners:

- `covered_rewrite`
  - a current-iteration write fully covers a carried region
- `partial_overwrite_remainder_read`
  - a carried region is only partially overwritten, and a later read observes the untouched remainder
- `mixed_role_regions`
  - one physical buffer has disjoint state and channel regions
- `unknown_effect`
  - an effectful statement-level call was seen, but the analysis does not have a region-level decoder for it

## Pattern Semantics

### `covered_rewrite`

This means a loop-carried definition from iteration `k - 1` is proven to be completely overwritten by iteration `k` before any unsupported remainder can survive.

This is a positive proof, not a hazard.

### `partial_overwrite_remainder_read`

This is a concrete witness:

1. a carried region exists
2. a current-iteration write only partially overlaps it
3. a later read touches the untouched remainder

This pattern is much stronger than “two regions partially overlap”. It means some carried subregion actually remains semantically visible.

### `mixed_role_regions`

This means one physical buffer contains both:

- a loop-carried region
- a disjoint iteration-local region

This is a factual partitioning result. Whether a downstream pass supports such a buffer is a separate policy decision.

### `unknown_effect`

This means the analysis encountered a statement-level opaque or update-state call whose memory effects are not decoded by the pass.

The analysis still returns a result for diagnostics, but the result is not complete enough to serve as a sound dependence oracle for transformations.

In practice, downstream passes should treat `unknown_effect` as a hard stop.

## How Downstream Passes Should Use The Result

The intended contract is:

1. Read `tl_buffer_dependency_analysis`
2. Reject the loop if `unknown_effect` is present
3. Use `edges`, `buffers`, and `patterns` to derive the planning policy

A typical downstream policy is:

- plain loop-carried buffers:
  - buffers with distance-1 `RAW`
  - no problematic patterns
- plain channel buffers:
  - buffers with only channel regions
  - no problematic patterns
- special handling:
  - `mixed_role_regions`
  - `partial_overwrite_remainder_read`
- hard reject:
  - `unknown_effect`

Python example:

```python
analysis = loop.annotations["tl_buffer_dependency_analysis"]

loop_carried_buffers = {
    edge.buffer.name
    for edge in analysis.edges
    if edge.dep_kind == "RAW" and int(edge.distance) > 0
}

special_handling = {
    pattern.buffer.name
    for pattern in analysis.patterns
    if pattern.kind in {"mixed_role_regions", "partial_overwrite_remainder_read"}
    and pattern.buffer is not None
}

has_unknown_effect = any(pattern.kind == "unknown_effect" for pattern in analysis.patterns)
if has_unknown_effect:
    raise RuntimeError("Dependency analysis is incomplete for this loop")
```

See `testing/python/transform/test_tilelang_transform_sunmmio_analyze_buffer_dependency.py` for direct examples of downstream-style consumption.

## Current Scope And Limits

The pass is designed for common lowered TileLang/Sunmmio TIR, not arbitrary TIR.

Current limits include:

- only `RAW` edges are modeled
- loop-carried distance is currently focused on `0` and `1`
- `While` handling is best-effort, not a full fixed-point loop solver
- `tvm_access_ptr` is modeled as full-buffer access
- unsupported statement-level opaque calls surface as `unknown_effect`

This is intentional. For supported kernels, the analysis should stay precise enough to guide pipeline planning. For unsupported shapes, it should fail closed instead of silently inventing a trustworthy-looking graph.

## Extending The Analysis

If a new effectful intrinsic becomes part of the supported lowering pipeline, update the call decoder in `src/analysis/analyze_pipelined_buffer_dependency.cc`.

If the intrinsic has explicit region operands, it should be modeled as concrete reads/writes. If it remains opaque, it will surface as `unknown_effect`, and downstream passes should reject the optimized path.

## Related Files

- `src/analysis/analyze_pipelined_buffer_dependency.cc`
- `src/analysis/pipelined_buffer_dependency.h`
- `src/analysis/pipelined_buffer_dependency.cc`
- `tilelang/transform/__init__.py`
- `testing/python/transform/test_tilelang_transform_sunmmio_analyze_buffer_dependency.py`