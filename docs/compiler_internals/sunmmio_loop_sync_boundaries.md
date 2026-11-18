# SunMMIO: wait at loop exits for proven-independent work

`InjectSunmmioSync` retains conservative loop-entry and backedge drains unless
one of two narrow proofs applies. There is no new frontend option. These
transformations do not construct a software pipeline or change collective
barriers and multicast completion.

Tracked in [issue #460](https://github.com/SUNMMIO/Tilelang/issues/460).

## Independent local maps: merge backedge drains

A serial outer loop can issue independent vector maps without draining
Vector/RSRAM at every backedge. The pass keeps the loop-entry drain, preserves
operation order, and appends one Vector/RSRAM completion at the outer exit.

The following is schematic lowered ordering, not parser-ready TIR:

```text
wait preceding producers
for group:
    tile load from local input
    elementwise computation
    tile store to a distinct output span
    # previously: wait Vector/RSRAM here
wait Vector/RSRAM
later scalar or DMA consumer of output
```

`IndependentTileLoopProof` requires:

- A static serial outer loop with nonnegative minimum, positive extent and
  unit step. Nested loops also have static bounds and unit step.
- All reads and writes use known local RSRAM allocations. No allocation that
  is written may be read anywhere in the accepted body.
- Writes occur inside tile loops and target a compact rank-1 buffer with zero
  element offset. Each index is affine in the outer induction variable, with
  a positive constant stride.
- The union of an iteration's write spans is smaller than its stride. The
  final write fits the allocation and the index's signed bit width.
- No unknown calls, control-flow statements, pointer effects, explicit
  synchronization, barriers, or pipeline/unknown loop annotations.

In-place maps, aliases, overlapping writes and unproved cases retain the
existing traversal. The motivating C4 initialization reads residual storage
and writes distinct output chunks; its stateful route accumulation is outside
this proof.

## Independent WSRAM DMA: defer the entry wait

A preceding ODMA0 DRAM-to-WSRAM transfer can remain outstanding across a tile
loop whose effects are confined to local RSRAM. The ODMA0 wait moves to that
loop's exit; all other entry and backedge waits remain unchanged.

```text
launch QK -> scores in RSRAM
launch V copy: DRAM -> WSRAM
wait TC and other existing producers needed by scores
tile loop: scale/mask/max on RSRAM
wait ODMA0                 # previously at the tile-loop entry
later PV consumes V
```

In the motivating C1 case, each core has FP32 `scores[32,256]` including padded
query rows, while a BF16 V page `[256,128]` (64 KiB) is copied to WSRAM. The
score loop does not access that transfer's source or destination.

`RsramOnlyTileLoopProof` requires:

- Every pending ODMA0 access is an asynchronous global/DRAM read or a write
  to a known local WSRAM allocation, with at least one of each. Completion is
  unit-wide, so the pass defers the entire eligible set or none of it.
- The crossed loop has lowered tile-domain annotations. Its explicit accesses
  use local RSRAM, including supported local reduction temporaries.
- Only reviewed arithmetic/select, nested supported loops, branches, local
  RSRAM allocations/declarations, and vector in-tile max/min/sum are accepted.
  A generic pure-call annotation does not prove hardware effects.
- WSRAM/DRAM accesses, external buffers, nested DMA, explicit synchronization,
  barriers, pointer escapes and unknown calls/annotations retain entry waits.

The deferred transfer is explicitly completed at this loop's exit, including
zero-trip and terminal loops. It never escapes the loop or allocation lifetime.
This proof does not establish iteration independence and does not remove
Vector/RSRAM backedge waits. Pipeline annotations are excluded from both proofs.

## Validation

Targeted positive and negative tests:

```bash
python -m pytest -q \
  testing/python/sunmmio/transform/test_independent_tile_loop_sync.py \
  testing/python/sunmmio/transform/test_loop_entry_dma_sync.py \
  testing/python/sunmmio/transform/test_inject_sync.py
```

Also run the existing SunMMIO transform, pipeline and codegen regressions.
Performance acceptance requires fresh, matched compiler/input/SDK runs and an
independent numerical oracle. Reduced wait counts alone do not establish saved
cycles: moving an ODMA0 wait can expose TC waits that were previously hidden.
