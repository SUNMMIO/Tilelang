# SunMMIO Pipeline Cost-Model Calibration Notes

## Status

An experimental gem5-aligned cost model was evaluated for the SunMMIO
pipeline planner and later removed from the default implementation. Its timing
constants were derived from a small set of isolated traces and were not broad
enough to represent different shapes, data types, memory paths, simulator
configurations, or future hardware revisions.

The production planner therefore uses the general heuristic cost model. The
calibration below is retained as reference data for a future target-specific,
configurable cost-model profile. It must not be treated as a universal hardware
contract.

## Experimental Model

All delays below are integral planner cycles.

### TensorCore

For an MMA with dimensions `(M, N, K)`, the experimental estimate was:

```text
work_blocks = ceil(M / 32) * ceil(N / 32) * ceil(K / 32)
delay       = 37 + work_blocks
```

The constants were fitted to isolated `tcStart -> Calc done` observations: 38
cycles for `32x32x32` and 41 cycles for `32x32x128`. The configured warm-up
before `tcStart` was excluded.

### ODMA

A transfer used the following formula:

```text
delay = first_access_latency + ceil(bytes / request_bytes)
        + completion_latency
```

| Source and destination path | First access | Request bytes | Completion |
| --- | ---: | ---: | ---: |
| DRAM to RSRAM | 67 | 1024 | 2 |
| DRAM to another memory | 67 | 1024 | 2 |
| RSRAM to WSRAM or ASRAM | 10 | 1024 | 2 |
| Any path involving TCM | 12 | 16 | 2 |
| Other paths | 7 | 1024 | 2 |

The DRAM-to-RSRAM choice used `dmaPreProc -> dmaEventDone` observations of 72
and 74 cycles for 2 KiB and 4 KiB transfers. Broadcast used a separate estimate:

```text
broadcast_delay = 52 + ceil(bytes / 512)
```

### VectorCore

The experimental analyzer counted expression-tree operations and charged the
following latency per 4096-bit vector chunk:

| Operation | Cycles |
| --- | ---: |
| Buffer load | 14 |
| Buffer store | 3 |
| Add, subtract, multiply | 4 |
| Min, max, comparison | 3 |
| Cast | 3 |
| `exp2` | 11 |
| Bitwise AND | 3 |
| In-tile sum reduction | 14 |
| Other supported in-tile reduction | 2 |

The load value came from isolated `vle16` samples `[13, 14, 14, 14, 13]`.
The store value rounded isolated samples `[3, 3, 3, 4, 4]` down to 3.

## Why It Was Not Kept as the Default

- The constants describe one simulator configuration rather than a target
  capability or a stable architectural rule.
- The TensorCore fit covers only two closely related BF16 MMA observations.
- Memory-path classification by buffer scope does not capture topology,
  contention, alignment, burst structure, or concurrent traffic.
- The VectorCore analyzer adds expression latencies serially and cannot model
  instruction overlap, issue width, reuse, or unsupported expressions.
- Constant loop extents and regions were required, rejecting valid dynamic TIR.
- Independent rounding and planner time scaling can amplify calibration error
  and change the selected initiation interval.

## Future Profile Requirements

A calibrated model can be reintroduced as an explicit SunMMIO target profile
when it provides:

1. Versioned parameters tied to a named hardware or simulator configuration.
2. Coverage across supported shapes, data types, alignments, and memory paths.
3. A fallback to the general heuristic model for unsupported or dynamic TIR.
4. Validation tests comparing predicted ordering and latency with measured
   traces over a representative kernel suite.
5. Configuration outside planner source code so calibration updates do not
   change scheduling logic.
