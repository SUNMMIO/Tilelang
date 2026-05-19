# SunmmioTileLoopFusion

This page explains `tl.transform.SunmmioTileLoopFusion`, the TIR pass that fuses lowered Sunmmio tile loops into shared loop prefixes.

After `LowerTilesLoop`, many Sunmmio kernels contain several adjacent `tile.scope_entry` regions that iterate over the same logical tile axes. If those regions remain separate, intermediate values often have to move from one region to the next through shared-memory materialization instead of staying alive inside a shared execution prefix. `SunmmioTileLoopFusion` discovers those opportunities, chooses a legal and profitable shared-loop structure, and rewrites the original statement interval into a fused loop tree.

The pass is specific to the lowered tile-loop form used by the Sunmmio backend. It is not a general-purpose loop fusion pass over arbitrary TIR.

## Why This Pass Exists

After tile lowering, it is common to see code like this:

```python
for i in T.serial(
    4,
    annotations={
        "tile.scope_entry": T.int32(1),
        "tile.execution_axis": T.int32(0),
        "tile.tile_size": [T.int32(8), T.int32(32)],
    },
):
    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
        Tmp_shared[i * 8 : i * 8 + 8, j * 32 : j * 32 + 32] = A_shared[
            i * 8 : i * 8 + 8, j * 32 : j * 32 + 32
        ]

for i in T.serial(
    4,
    annotations={
        "tile.scope_entry": T.int32(1),
        "tile.execution_axis": T.int32(0),
        "tile.tile_size": [T.int32(8), T.int32(32)],
    },
):
    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
        B_shared[i * 8 : i * 8 + 8, j * 32 : j * 32 + 32] = Tmp_shared[
            i * 8 : i * 8 + 8, j * 32 : j * 32 + 32
        ]
```

These two regions touch the same logical execution prefix (`i`, then `j`) and the second region consumes the first region's output tile by tile. Leaving them separate has three costs:

- Producer/consumer reuse is cut between regions.
- More data must be materialized and later re-read.
- The compiler misses the chance to keep values resident inside a shared loop prefix.

The fused shape is conceptually:

```python
for i in T.serial(
    4,
    annotations={
        "tile.scope_entry": T.int32(1),
        "tile.execution_axis": T.int32(0),
        "tile.tile_size": [T.int32(8), T.int32(32)],
    },
):
    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
        Tmp_shared[i * 8 : i * 8 + 8, j * 32 : j * 32 + 32] = A_shared[
            i * 8 : i * 8 + 8, j * 32 : j * 32 + 32
        ]
        B_shared[i * 8 : i * 8 + 8, j * 32 : j * 32 + 32] = Tmp_shared[
            i * 8 : i * 8 + 8, j * 32 : j * 32 + 32
        ]
```

That shared loop prefix is the central goal of the pass.

---

## Where It Runs In The Pipeline

`SunmmioTileLoopFusion` expects already-lowered Sunmmio tile-loop TIR. In practice it runs after `LowerTileOp` and `LowerTilesLoop`.

```python
import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC

target = tvm.target.Target(SUNMMIO_TARGET_DESC)
with tvm.target.Target(target):
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl.transform.AddWrapperForSingleBufStore()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.InferSramScope()(mod)
    mod = tl.transform.LayoutReducer()(mod)
    mod = tl.transform.LayoutInference()(mod)
    mod = tl.transform.LowerTileOp()(mod)
    mod = tl.transform.LowerTilesLoop()(mod)

mod = tl.transform.SunmmioTileLoopFusion()(mod)
```

If you apply the pass before the IR contains `tile.scope_entry` regions, it usually has nothing useful to fuse.

---

## Core Terms

- **Planner-visible region**
  One lowered tile region rooted at a `tile.scope_entry` loop, together with any wrapper structure that still belongs to that region. Discovery also recognizes thin `AttrStmt`, `LetStmt`, `Block`, and `BlockRealize` wrappers around the scope-entry loop, so regions with surrounding local bindings or annotations are still discovered as single units.

- **Execution loops**
  The loops directly under the `tile.scope_entry` that carry `tile.execution_axis` metadata. These are the loops the pass may choose to share.

- **Logical execution axis key**
  A canonical axis label such as `i`, `j`, or `k`. Discovery maps lowered loop variables into this logical space so two regions can still be compared even if their concrete loop var names differ. The mapping is derived from `tile.execution_domain_axes` annotations on the scope-entry loop: each execution loop's `tile.execution_axis` index is resolved through this domain-axis array to produce a stable logical label.

- **Planning group**
  A contiguous group of adjacent planner-visible regions that the pass analyzes, plans, and rewrites together. If any unrelated statement appears between two regions, they belong to different groups. In the implementation, this group is later packaged into planner-facing `WindowProblem` and `WindowPlan` objects, but you can read the pass as operating on one group at a time.

- **`use_in`**
  External buffer regions the region reads when it begins execution. Local scratch buffers and planner-private scopes (any scope other than `global` or `shared*`) are filtered out before they reach the planner.

- **`def_out`**
  External buffer regions the region produces. The same filtering rules apply.

- **Availability depth / `home_depth`**
  These are two stage-specific names for the same idea: the shallowest shared execution-loop depth at which a value or access no longer depends on any deeper loop indices. For example, if a produced value still changes with both `i` and `j`, it needs depth 2. If it changes with `i` but not with `j`, depth 1 is enough.

- **Dependence kinds**
  `RAW` means a later region reads a value produced by an earlier one.
  `WAR` means a later region overwrites something an earlier region still reads.
  `WAW` means two regions write overlapping output.

- **`rho`**
  The minimum shared execution depth required to keep a dependence internal. If `rho = 1`, sharing only the outer execution loop is enough. If `rho = 2`, the producer and consumer must also share the next inner loop. `rho` is computed from the deepest execution-loop depth referenced by either the source or destination normalized access dimensions.

- **Shared loop prefix**
  The outer execution loops that multiple regions execute inside after fusion. In implementation comments and helper names, this is sometimes called a "shared shell."

For example, these two separate loops:

```python
for i in ...:
    region_A_for_i()

for i in ...:
    region_B_for_i()
```

can become:

```python
for i in ...:
    region_A_for_i()
    region_B_for_i()
```

After fusion, the shared loop prefix is just the outer `for i in ...` loop. If both `i` and `j` are shared, then the shared loop prefix is the nested `for i` / `for j` prefix.

- **Leaf**
  A concrete region scheduled under some shared loop prefix. During rewrite, a leaf keeps only the private execution suffix below the shared depth.

---

## High-Level Pipeline

`SunmmioTileLoopFusion` is intentionally staged:

```text
lowered PrimFunc
      |
      v
SunmmioTileLoopFusionProgram
  - planner-visible regions
  - planning groups of adjacent regions
      |
      v
SunmmioTileLoopFusionWindowProblem
  - normalized external reads/writes
  - group-local dependence graph
      |
      v
SunmmioTileLoopFusionWindowPlan
  - chosen region order
  - shared loop prefix tree
      |
      v
rewritten PrimFunc
  - original interval replaced by fused loop-prefix tree
```

1. `PrimFunc -> SunmmioTileLoopFusionProgram`
   Discovery finds every planner-visible region and partitions them into planning groups of adjacent regions.
2. `Program -> SunmmioTileLoopFusionWindowProblem`
   For each group, the regions' external reads and writes are normalized into logical-axis space and a group-local dependence graph is built.
3. `WindowProblem -> SunmmioTileLoopFusionWindowPlan`
   The planner chooses a shared-loop structure and execution order for each group.
4. `WindowPlan -> rewritten PrimFunc`
   Rewrite replaces the original statement interval with the fused loop-prefix tree.

Conceptually, the pass solves a small graph problem for each planning group:

- Stage 1 identifies the regions that may participate. You can think of these as the graph nodes.
- Stage 2 determines which regions exchange or conflict on data. You can think of these as the graph edges.
- Stage 3 chooses an execution order and shared-loop structure that respects and profits from that graph.
- Stage 4 emits the chosen structure back into TIR.

This split matters because each stage owns a different concern:

- Discovery understands lowered TIR shape.
- Normalization and dependence construction understand buffer overlap.
- Planning understands legality and profitability.
- Rewrite only rebuilds TIR from the chosen plan.

---

## Stage 1: Discovery

Discovery walks the lowered `PrimFunc` and looks for planner-visible regions. At the end of this stage, the pass knows the candidate region nodes for each planning group, but it does not yet know how those regions relate to each other. Each discovered region records:

- The original wrapped root statement.
- The actual `tile.scope_entry` loop.
- The exposed execution loops under that scope entry.
- Canonical logical axis keys for those loops.
- The execution extents of the exposed loops.
- External reads (`use_in`) and writes (`def_out`).
- The execution depth where each produced value becomes available.

### What counts as one region

The pass does not only remember the bare `tile.scope_entry` loop. It keeps the full wrapped statement that owns that region. This is why rewrite can later preserve local wrappers such as `LetStmt` nodes and can hoist compatible `AttrStmt` wrappers. Discovery also recognizes `BlockRealize` and `Block` wrappers (provided the block has no `init`), so regions that lowering wraps in a block boundary are still matched as single planner-visible units.

### How planning groups are formed

Discovery partitions the program into contiguous groups of adjacent planner-visible regions. Each group is one exact interval that the pass may plan and rewrite together. The pass does not pretend unrelated statements are part of the same planning problem. Discovery walks enclosing `SeqStmt` nodes: for each child, it attempts to match a planner-visible region. Consecutive matches form a run; any unrecognized child flushes the current run and starts a new one.

### How a region is summarized for planning

Before the planner can decide whether two regions should be fused, it needs a compact summary of what can cross from one region to another. For each region, discovery therefore keeps only the region's external interface:

- which buffer slices the region reads from outside itself
- which buffer slices the region writes for later regions to observe

In the implementation, those two parts are named `use_in` and `def_out`. You can read them as "external reads" and "external writes."

For example, consider a region like:

```python
for i in ...:
    for j in ...:
        Tmp_shared[i, j] = A_shared[i, j]
```

For planning purposes, the important summary is simply:

- external reads: `A_shared[i, j]`
- external writes: `Tmp_shared[i, j]`

That is the information later stages use to answer questions such as the following. These per-region summaries are exactly the inputs Stage 2 will compare when it builds the group's data-flow/dependence graph.

- Does a later region read something this region produced?
- Do two regions touch overlapping data?
- At what shared-loop depth does a produced value become reusable?

Discovery builds this summary from the body below the exposed execution loops, with a few simplifying rules:

- Local scratch buffers (allocated inside the region via `alloc_buffer` or `BufferRealize`) are ignored, because other regions cannot observe them.
- Planner-private buffers (scopes other than `global` or `shared*`) are filtered out.
- Duplicate external accesses are deduplicated.
- Write-only opaque accesses are removed from the external-read set.
- Opaque intrinsic calls (such as `vector_core_in_tile_reduce`) are handled specially: discovery extracts their read/write buffer regions and merges them into the standard `use_in`/`def_out` sets.
- For each external write, discovery also records how much of the surrounding loop nest must be fixed before that result is determined.

That last point matters because a region can have two execution loops and still produce a value that depends only on the outer loop variable. In that case the result becomes reusable after the outer loop is fixed, rather than only after both loops are fixed.

---

## Stage 2: Normalization And Dependence Construction

Stage 1 summarized each region independently. Stage 2 is where the pass connects those isolated region summaries into the actual planning problem for one planning group. If Stage 1 gives the nodes, Stage 2 gives the edges.

At a high level, this stage answers two questions:

- Which regions touch the same logical data, even if lowering gave them different loop-variable names?
- When two regions do touch the same logical data, what dependence does that create, and how much shared-loop depth is needed to keep that dependence internal?

The output of Stage 2 is a group-local data-flow/dependence graph over the regions in one planning group.

### 1) Put all accesses in one logical coordinate system

The first problem is comparison. Two regions may mean the same logical access while using different lowered loop-variable names.

For example, one region may contain:

```python
Tmp_shared[i0 * 8 + ki, j0 * 32 + kj]
```

while another region contains:

```python
Tmp_shared[i1 * 8 + ki, j1 * 32 + kj]
```

If the pass compared those accesses literally, it would treat them as different just because one region uses `i0, j0` and the other uses `i1, j1`.

Normalization fixes that by rewriting each external read and write into one shared logical coordinate system with canonical axis names such as `i`, `j`, and `k`. A single set of canonical `Var` objects is shared across all regions in the program, so that structurally equal expressions in different regions truly compare as equal.

After normalization, both examples above become:

```python
Tmp_shared[i * 8 + ki, j * 32 + kj]
```

Now the two regions can be compared by what part of the logical tile space they access, rather than by the accidental variable names produced during lowering.

### 2) Build a normalized record for each external read and write

Once an external read or write has been rewritten into that shared logical coordinate system, the pass stores it as a small planner-facing record. The goal is not to keep all of TIR around; it is to keep exactly the facts needed for dependence construction and planning.

Each normalized access record contains:

- the normalized `BufferRegion`, which says what logical slice of the buffer is touched
- per-dimension `min` and `extent`, which say exactly where that slice starts and how large it is
- the execution-loop depths each dimension depends on, which say how much of the loop nest must be shared for the access to line up across regions
- a `home_depth` and payload size in bytes, which say when the value becomes reusable and how expensive it is to keep or rematerialize

You can think of this record as combining three kinds of information in one place:

- geometry: what data is touched
- loop dependence: which execution loops that access still depends on
- reuse: after what shared-loop depth the value can be treated as available

For reads, `home_depth` is inferred from the deepest execution loop referenced by the access. When no execution loop is referenced but the region has execution loops, `home_depth` falls back to the minimum of the region's execution rank and the access's buffer-region rank. For writes, discovery can set a shallower `home_depth` when the produced result no longer depends on the deepest loop, which means later regions may be able to reuse it earlier.

### 3) Compare normalized accesses to build dependence edges

Now that every region's external reads and writes are expressed in the same logical coordinate system, the pass can compare them across regions in the same planning group.

Comparison uses per-dimension range intersection. Two normalized accesses to the same buffer are intersected dimension by dimension: each dimension computes `max(lhs_min, rhs_min)` as the intersection start and `min(lhs_end, rhs_end)` as the intersection end. If any per-dimension intersection extent is provably non-positive (using TVM's arithmetic analyzer), the accesses do not overlap and no dependence edge is created. If all dimensions have a non-empty intersection, an edge is created.

If they do overlap, the pass creates one of three edge kinds:

- `RAW`: a later region reads a value produced by an earlier region
- `WAR`: a later region overwrites something an earlier region still reads
- `WAW`: two regions write overlapping output

This is the key transition in Stage 2: the pass moves from isolated per-region summaries to a group-level data-flow/dependence graph that the planner can actually schedule.

### 4) Annotate each edge with what the planner needs to know

The planner needs more than just "these two regions depend on each other." It also needs to know how strong that dependence is.

Each edge therefore carries two important extra facts:

- `rho`: the minimum shared-loop depth required to keep that overlap internal
- `weight`: an estimated byte cost for cutting the dependence

#### What `rho` means

`rho` tells the planner how much of the outer loop nest must be shared before the producer and consumer refer to the same value instance.

- If the overlap varies only with the outer execution loop, then `rho = 1`.
- If the overlap varies with both the outer and next inner execution loops, then `rho = 2`.

So `rho` is not just "there is a dependence." It is "how deeply do these regions need to share loops for that dependence to stay internal?"

#### What edge weight means

Today only `RAW` edges carry a direct cut cost. The pass estimates that cost in bytes of overlap payload that would have to leave the fused loop prefix if the dependence were cut. The current model treats that as a spill/reload pair, so the byte cost is effectively doubled (the implementation computes `2 * element_count * dtype_bytes`). When the exact overlap extent is not a constant, the pass falls back to the minimum of the source and destination per-dimension static extents.

`WAR` and `WAW` edges still matter for legality, but they do not currently contribute a direct cut-byte term (their weight is always 0).

### 5) Active-set sweep for graph construction

Edges are not built by comparing every pair of accesses across every pair of regions. Instead, the implementation sweeps through the planning group in source order, maintaining active sets of the most recent definitions and uses per buffer.

For each new region:
1. Its reads are compared against active definitions to find `RAW` edges.
2. Its writes are compared against active reads to find `WAR` edges. Overlapping earlier reads are killed from the active set.
3. Its writes are compared against active definitions to find `WAW` edges. Overlapping earlier definitions are killed from the active set.
4. The new region's reads and writes are added to the active sets.

This sweep-based approach means that a later overwrite of a buffer kills the earlier definition from the active set, so only the latest relevant producer/consumer relationships are tracked. For example, if regions A, B, and C all write the same buffer in sequence, only B→C edges are created (not A→C), because B's write kills A's definition.

After Stage 2, the planner no longer reasons about raw TIR accesses. It reasons about regions connected by a data-flow/dependence graph whose edges already encode overlap kind, required shared-loop depth, and estimated cut cost.

---

## Stage 3: Planning

Stage 2 built a data-flow/dependence graph for one planning group. Stage 3 chooses a schedule for that graph.

Here, "schedule" means two things at once:

- the order in which the regions will execute
- the shared loop-prefix structure they will execute inside

The output of Stage 3 is therefore not just a reordered list of regions. It is a rewrite-ready fused loop-prefix tree.

Before looking at the implementation details, it helps to have the high-level dynamic-programming picture in mind. The planner does not pick the whole schedule at once. Instead, it builds the plan one region at a time.

Each dynamic-programming state describes one partial plan:

- which regions have already been scheduled
- which shared loop prefixes are currently open
- which produced values are still resident under those open prefixes

One transition extends that partial plan by choosing one next region and the shared-loop depth where it should attach. In other words, one step decides both "who runs next" and "inside how much shared loop structure it runs."

This is a natural fit for dynamic programming because many different action sequences can lead to the same partial state. Once the planner has computed the best continuation from that state, it can reuse that answer anywhere else the same state appears.

### 1) What the planner receives and what it produces

The planner starts from one planning group whose regions are already connected by a data-flow/dependence graph. From that input, it must produce a plan that is both legal and profitable.

A typical output looks like this:

```text
scope(i)
  scope(i, j)
    region_0
    region_2
  region_1
```

This means:

- every child shares the outer `i` loop
- `region_0` and `region_2` also share the deeper `i, j` prefix
- `region_1` stays only under the outer prefix

That tree is the planner's real product. Stage 4 will later turn it back into TIR.

### 2) What the planner must decide

For each planning group, the planner is trying to answer three questions:

- Which legal region should run next?
- How much of the current shared loop prefix should stay open?
- Should the next region attach at the root, under the outer loop only, or under a deeper shared prefix?

So the planner is not solving only an ordering problem. It is solving an ordering-plus-loop-sharing problem.

### 3) Why the planner state is more than just region order

It is not enough to know which regions have already been scheduled. Future decisions also depend on what reuse is still available at the current point in the schedule.

For that reason, the planner state tracks three things:

- which regions have already been scheduled (as a dynamic bitset)
- which shared loop prefixes are currently open (as a stack of scope frames)
- which values are still resident and visible inside those open prefixes

Those resident values are what let the planner reason about reuse. A later region may be able to consume a producer's result directly if that result is still resident under a sufficiently deep shared loop prefix. Resident values come in two kinds: `kDefinition` (a producer's output that a later RAW consumer may reuse) and `kRead` (a materialized read that a later region sharing the same buffer access may reuse).

### 4) What one planner action means

One planner action chooses one next region and two depths:

- `close_to_depth`: keep the first `N` currently open shared loop prefixes and close anything deeper
- `open_to_depth`: open any additional shared loop prefixes for the chosen region until the depth where it should attach

This lets the planner express choices like:

- attach the next region at the root with no shared loop prefix (`close_to_depth=0, open_to_depth=0`)
- keep sharing only the outer execution loop (`close_to_depth=1, open_to_depth=1`)
- extend the current prefix to share both outer and inner loops (`close_to_depth=1, open_to_depth=2`)

In other words, one action says both "run this region next" and "run it under this amount of shared loop structure."

### 5) How the planner evaluates one action

When the planner considers one action, it computes the immediate cost of that choice and the resulting next state.

That evaluation does five important things:

1. Rebuild the open shared loop-prefix stack implied by `close_to_depth` and `open_to_depth`.
2. Check incoming `RAW` edges. If the required producer definition is not visible at this attach depth, that dependence is cut and contributes `write_cut_cost`. If the producer definition is visible, the covered use is marked so it does not also contribute shared-read cost.
3. Charge `shared_read_cost` for uncovered reads that are not already satisfied by a visible resident value (either a definition or a read resident).
4. Update resident values: kill all residents for any buffer that the current region overwrites (by buffer name), then install newly available definitions. Outgoing RAW edges are also walked to pre-install producer definitions at their dependence `rho` depth.
5. Mark the region as scheduled, prune residents that have no remaining future consumers, compute the remaining live resident footprint (`live_range_penalty`), and count source-order inversions (`reorder_penalty`).

This is why the planner needs the full state described above. The immediate cost of a choice depends on what values are still visible, not just on which regions remain unscheduled.

### 6) How the planner compares candidates

The planner uses a lexicographic score rather than collapsing everything into one weighted scalar.

The score terms are, in priority order:

1. `write_cut_cost`
   Estimated bytes paid when a `RAW` overlap escapes the fused loop prefix.
2. `shared_read_cost`
   Estimated bytes materialized for uses that are not already covered by visible residents.
3. `live_range_penalty`
   Estimated bytes kept live across the currently open shared loop prefixes, weighted by each resident's execution-prefix instance count.
4. `reorder_penalty`
   Count of earlier source-order regions skipped over by the chosen action.

This priority order is strict. A one-byte improvement in `write_cut_cost` dominates any change in the lower-priority terms. All score arithmetic uses saturating add/multiply (clamped to `int64_t::max / 4`) to prevent overflow.

### 7) How the planner searches

For small planning groups (up to 15 regions), the planner uses a memoized exact search:

```text
best(state) = min_a delta(state, a) + best(next_state(state, a))
```

The memo key is the full planner state serialized as a string, not just the scheduled-region mask, because future costs depend on both the open shared loop prefixes and the resident values attached to them.

The search also applies a pruning bound: because all score terms are nonnegative, any branch whose immediate transition delta already exceeds the best complete plan found so far can be skipped without recursion.

To keep compile time bounded, two hard limits are enforced:

- **Region count limit**: groups with more than 15 regions skip exact search entirely and use source-order fallback.
- **Memo table limit**: if the memo table reaches 200,000 entries, the exact search is abandoned and the caller falls back to source-order planning.

The source-order fallback schedules each region at the root depth (no shared loop prefix) in the first legal source-order position, which is conservative but guaranteed to terminate quickly.

After Stage 3, the pass has chosen one concrete fused loop-prefix tree for the planning group. Stage 4 is then responsible for materializing that tree back into TIR.

---

## Stage 4: Rewrite

Stage 3 chose a fused loop-prefix tree for one planning group. Stage 4 turns that abstract plan back into concrete TIR.

At a high level, rewrite is a structural reconstruction step, not another planning step. The planner has already decided:

- which regions belong together
- what order they should execute in
- how much loop prefix they should share

Rewrite's job is to emit new TIR that realizes exactly that decision while preserving the original region bodies and their local structure.

### 1) What rewrite receives and what it produces

For one planning group, rewrite receives two things:

- the original contiguous statement interval from the lowered `SeqStmt`
- the planner's fused loop-prefix tree for that interval

Its output is one replacement TIR subtree for that same interval.

You can think of Stage 4 as translating from a planner view back to a TIR view:

```text
planner tree:
  scope(i)
    scope(i, j)
      region_0
      region_2
    region_1

becomes TIR shaped like:
  for i in ...:
    for j in ...:
      body(region_0)
      body(region_2)
    body(region_1)
```

So the essential question in this stage is simple: how do we rebuild concrete loops and region bodies so that they match the tree the planner chose?

### 2) Converting the action trace to a tree

The planner produces a linear action trace (a sequence of `close_to_depth` / `open_to_depth` / `region_index` triples). Before rewrite can use this, the trace is replayed against a mutable stack of scope nodes to reconstruct the nested tree structure. Each `close_to_depth` pops the stack back, each `open_to_depth` pushes new scope nodes, and the region is attached as a leaf under the current top of stack. The result is then frozen into an immutable `SunmmioTileLoopFusionPlannerTreeNode` tree.

### 3) Replace one exact statement interval at a time

Rewrite does not search the whole function again. Each planned group remembers the exact contiguous statement interval it came from, and rewrite only substitutes when the current `SeqStmt` still contains that same interval unchanged. Matching uses pointer identity (`same_as`), so it is precise and fast.

This keeps the transformation local and precise:

- the pass rewrites only the statements that were actually planned together
- unrelated surrounding statements are left untouched
- the emitted subtree drops back into the original function at the same place

### 4) Rebuild each shared loop prefix from a representative region

Each internal scope node in the planner tree says that several children should execute under one shared loop prefix. Rewrite materializes that scope by choosing one representative region under the scope and reusing that region's original loop structure for the shared prefix.

Concretely, rewrite:

- picks the first leaf region under the scope as the representative
- takes that region's original execution loops up to the shared depth
- creates fresh `Var` objects for the shared loop variables (preserving the original name hints)
- rebuilds those loops around the rewritten children of the scope node, substituting the representative region's original loop variables with the outer shared vars where needed

This is legal because the planner only groups regions under the same shared loop prefix when their shared axes and extents already match.

### 5) Turn each planned region into a leaf under that prefix

A planned region usually contains both:

- the part of its loop nest that is now shared with siblings
- the part that remains private to that region

When rewrite places that region under a shared loop prefix, it removes the already-shared execution prefix from the region and keeps only the remaining private suffix and body.

So if the planner says that two regions share `for i in ...:` but not any deeper loop, rewrite does not duplicate that `i` loop inside each leaf. Instead, it emits one outer `i` loop and puts each region's private remainder inside it.

To make the rebuilt prefix line up with the original body, rewrite substitutes the shared loop variables from the rebuilt loops into the leaf body. This substitution uses a `ScopeEntryReplacer` mutator that finds the original `tile.scope_entry` loop in the (potentially wrapped) region root and replaces it with the rebuilt private suffix.

### 6) Preserve wrappers and local structure

The rebuilt tree should preserve the important structure that originally surrounded each region.

Rewrite therefore treats wrappers carefully:

- Common leading `AttrStmt` wrappers that are compatible with the shared loop prefix can be hoisted once around the fused subtree. Two `AttrStmt` frames are considered "the same" when they agree on `attr_key`, `node`, and `value` (checked by structural equality). Only the longest common prefix across all leaf regions is hoisted. `AttrStmt` wrappers that reference execution-loop variables are never hoisted.
- Local `LetStmt` wrappers stay attached to the leaf that originally owned them, because they may bind values specific to that region.
- `AttrStmt` hoisting is cumulative through scope nesting: each scope node additionally hoists any common `AttrStmt` prefix shared by its children, beyond what the parent scope already hoisted.

This is why rewrite can both simplify the outer structure and still preserve region-local bindings and annotations.

After Stage 4, the planner's abstract loop-prefix tree has been turned back into ordinary TIR. The result is a local replacement that realizes the chosen fusion structure without changing the meaning of the surrounding function.

---

## Worked Example: Two Consecutive Tile Copies

This is the smallest non-trivial case covered by the rewrite tests.

Before fusion:

```python
for i in T.serial(
    4,
    annotations={
        "tile.domain": [T.int32(32), T.int32(32)],
        "tile.execution_axis": T.int32(0),
        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
        "tile.scope_entry": T.int32(1),
        "tile.tile_size": [T.int32(8), T.int32(32)],
    },
):
    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
        Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj]

for i in T.serial(
    4,
    annotations={
        "tile.domain": [T.int32(32), T.int32(32)],
        "tile.execution_axis": T.int32(0),
        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
        "tile.scope_entry": T.int32(1),
        "tile.tile_size": [T.int32(8), T.int32(32)],
    },
):
    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
        B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj]
```

Discovery sees two regions in one planning group. Dependence construction creates one `RAW` edge from the `Tmp_shared` producer to the `Tmp_shared` consumer. The required shared depth is `rho = 2`, so a shared loop prefix that includes both outer and inner execution loops is needed to keep the overlap internal.

After fusion, the pass emits one shared loop prefix whose body contains both leaves:

```python
for i in T.serial(
    4,
    annotations={
        "tile.domain": [T.int32(32), T.int32(32)],
        "tile.execution_axis": T.int32(0),
        "tile.execution_domain_axes": [T.int32(0), T.int32(1)],
        "tile.scope_entry": T.int32(1),
        "tile.tile_size": [T.int32(8), T.int32(32)],
    },
):
    for j in T.serial(1, annotations={"tile.execution_axis": T.int32(1)}):
        Tmp_shared[i * 8 + ki, j * 32 + kj] = A_shared[i * 8 + ki, j * 32 + kj]
        B_shared[i * 8 + ki, j * 32 + kj] = Tmp_shared[i * 8 + ki, j * 32 + kj]
```

The important change is not just that the loops look shorter. The producer and consumer are now siblings under the same shared loop prefix, so the producer/consumer reuse stays internal to the shared loop prefix.

---

## Worked Example: Different Depths Of Sharing

The planner is not limited to "all or nothing" fusion.

### Sharing Only The Outer Loop

If a producer emits a value that depends only on the outer loop `i`, and the consumer only needs that per-`i` value, the edge may have `rho = 1`. In that case the planner can build only a depth-1 shared loop prefix:

```text
scope(i):
  region_0
  region_1
```

### Nested tile sharing

If one consumer needs reuse only after both `i` and `j` are fixed, while another needs reuse only after `i` is fixed, the planner may legally reorder the consumers and build a nested tree:

```text
scope(i):
  scope(i, j):
    region_0
    region_2
  region_1
```

This kind of plan appears in the planner tests. It keeps the deeper `i,j` consumer close to the producer while still preserving a shallower shared loop prefix that shares only the outer loop for the other consumer. In this case, `region_2` (originally third in source order) is reordered ahead of `region_1` because the tile-level RAW dependence (with higher byte weight) takes priority over the row-level dependence.

### Real-kernel cases

The Python rewrite tests also cover larger structural examples:

- Flash-attention online softmax groups, where some operations fuse at depth 1 (row-level: `scores_max` + `scores_scale`) and others at depth 2 (tile-level: `acc_s` + `acc_s_cast` + reduction).
- RMSNorm groups, where elementwise work (`a_square`) and row reductions (`reduce_row_sum`) are fused at depth 2 while reduction-local scratch allocations remain correctly scoped inside their `Block` wrappers.

Those tests are useful references because they show the pass operating on realistic lowered kernels rather than only hand-built unit cases.

---

## What The Pass May Change

- It may fuse multiple adjacent `tile.scope_entry` regions into one shared loop prefix.
- It may build nested shared loop prefixes instead of only a single outer shared loop prefix.
- It may reorder regions inside one planning group when the dependence graph allows it and the planner score improves.
- It may hoist a common leading `AttrStmt` prefix around the fused loop prefix.
- It may leave some regions as standalone leaves at the root if sharing them is not legal or not profitable.

## What The Pass Does Not Do

- It does not fuse arbitrary TIR loops that are not lowered Sunmmio tile regions.
- It does not fuse across unrelated statements outside a discovered planning group.
- It does not ignore `RAW`, `WAR`, or `WAW` hazards.
- It does not promise global optimality once the planner falls back to source order for large planning groups.
- It does not remove local wrappers such as region-specific `LetStmt` bindings.
- It does not hoist `AttrStmt` wrappers that reference execution-loop variables.

---

## Common Reasons Fusion Is Limited Or Absent

- **Only one region in the planning group**
  A single planner-visible region is already maximal for that interval, so the pass is effectively a no-op.

- **Incompatible shared prefix**
  Two regions may touch related buffers but still disagree on execution axes or extents, so they cannot be placed under the same shared loop prefix. The planner checks both axis labels and extents (using structural equality) at each depth.

- **Insufficient legal sharing depth**
  An edge may require a deeper shared prefix than the candidate regions can legally share.

- **No profitable resident reuse**
  The planner may decide a more fused shared loop prefix would increase higher-priority score terms instead of reducing them.

- **Large planning group**
  Groups with more than 15 regions fall back to a conservative source-order plan to keep compile time bounded. Even smaller groups may fall back if the memoized state space exceeds 200,000 entries.

- **No overlapping external reads/writes**
  If the normalized read/write regions do not overlap, no dependence edge is created, so there may be nothing useful to internalize.

---

## Implementation Files

| File | Stage | Purpose |
|------|-------|---------|
| `src/transform/sunmmio_tile_loop_fusion/types.h` | All | Shared stage-boundary data structures |
| `src/transform/sunmmio_tile_loop_fusion/cost_model.h/cc` | 3 | Lexicographic score representation and saturating arithmetic |
| `src/transform/sunmmio_tile_loop_fusion/utils.h/cc` | 1–2 | Logical-axis normalization, `VarUseCollector`, expression helpers |
| `src/transform/sunmmio_tile_loop_fusion/discovery.h/cc` | 1–2 | Region discovery, buffer access analysis, normalization, and dependence graph construction |
| `src/transform/sunmmio_tile_loop_fusion/planner.h/cc` | 3 | Public planning entrypoint and input preprocessing |
| `src/transform/sunmmio_tile_loop_fusion/planner_internal.h` | 3 | Private solver types: bitsets, resident state, memo/search records |
| `src/transform/sunmmio_tile_loop_fusion/planner_solver.cc` | 3 | Exact DP search, source-order fallback, and plan tree reconstruction |
| `src/transform/sunmmio_tile_loop_fusion/pass.cc` | 4 + entry | Rewrite mutator and module-pass entry point |

---

## How To Inspect The Result

To see what the pass changed, print the lowered module before and after the pass:

```python
mod = tl.transform.LowerTilesLoop()(mod)
print("=== Before Fusion ===")
print(mod.script())

mod = tl.transform.SunmmioTileLoopFusion()(mod)
print("=== After Fusion ===")
print(mod.script())
```

For Sunmmio kernels, a more realistic sequence is to run the full lowering pipeline through `LowerTileOp` and `LowerTilesLoop`, then inspect the rewritten module after fusion.

Useful regression tests include:

- `testing/python/transform/test_tilelang_transform_sunmmio_tile_loop_fusion_rewrite.py`
- `testing/cpp/transform/sunmmio_tile_loop_fusion/discovery_test.cc`
- `testing/cpp/transform/sunmmio_tile_loop_fusion/planner_test.cc`
- `testing/cpp/transform/sunmmio_tile_loop_fusion/planner_internal_test.cc`
- `testing/cpp/transform/sunmmio_tile_loop_fusion/cost_model_test.cc`

---

## FAQ

- **Is the pass always a net win?**
  The pass is heuristic and compile-time only. It tries to preserve the most important reuse first, but it is not a calibrated runtime cost model.

- **Can the pass reorder regions?**
  Yes. Reordering is allowed inside a planning group when dependence legality is preserved and the lexicographic planner score improves. Reordering is only a tie-breaker after the three byte-based score terms.

- **Why does one case share only the outer loop while another shares both outer and inner loops?**
  Because `rho` depends on which execution depths the overlapping reads and writes actually vary with.

- **Why are some wrappers hoisted while others stay local?**
  Common leading `AttrStmt` wrappers that do not depend on execution-loop vars can be hoisted. Local `LetStmt` wrappers remain attached to the leaf that owns them because they may bind region-specific values.

- **Why is a large example still close to source order after fusion?**
  The planner intentionally falls back to a conservative source-order plan once the exact search group becomes too large or the memoized state space is exhausted.

- **What happens if two different buffers share the same name?**
  The planner kills resident values by buffer name string, not by buffer identity. In practice this is safe because lowered buffer names are unique, but it is an implementation assumption worth knowing about.
