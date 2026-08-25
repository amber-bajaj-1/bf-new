# Phase 13 Workspace Decision

## Decision status

The Phase 13 decision is **provisional and bounded-synthetic only**:

```text
vertex storage: full graph-sized vertex-major lanes
run storage: retained per-run lane-mask arrays with touched ledgers
production status: deferred
```

This choice is not a GPU recommendation. It selected the simpler host/device
contract consumed provisionally by the Phase 14, 15, and 16 first device
paths while capacity appeared plausible under an explicitly illustrative
calculation. The real `logicnets_jscl` `.bfqueries` artifact, real batch union,
active-run, and frontier distributions, runtime free-device-memory observation,
HIP compilation, and device execution are absent. Any of those measurements
may reverse the production choice.

Phase 13 itself implements no HIP source and performs no device allocation.

## What was compared

Every comparison uses the same validated plan and exact batch description. The
model evaluates this cross product:

| Vertex-label storage | Run-admission storage | Additional structure |
| --- | --- | --- |
| Full graph vertex-major | Retained per-run masks | Dense CSR/CSC masks plus touched ledgers |
| Full graph vertex-major | Compact nonzero descriptors | Sorted `(run_id,lane_mask)` records and descriptor offsets |
| Compact union tiles | Retained per-run masks | Dense per-tile global-to-compact bias mapping |
| Compact union tiles | Compact nonzero descriptors | Bias mapping plus sorted descriptors and per-vertex offsets |

Full storage allocates `graph_vertex_count * lane_width` label positions for
each distance slot. Compact storage allocates only union-tile vertices and
retains one `uint32_t` bias per tile. For a selected endpoint, the packed index
is `global_vertex - bias[owner_tile]`, guarded by its tile lane mask; the bias
is `global_tile_begin - packed_tile_begin`. Retained run storage keeps a mask
slot for the larger of the CSR/CSC run counts and clears only positions named
by the prior touched ledger. Descriptor storage writes only nonzero runs but
requires compact records and per-vertex descriptor offsets for traversal.

The checked report keeps these quantities separate:

- distance-label, tile-mapping, run-storage, descriptor-offset, and batch-
  metadata allocation bytes;
- selected, allocated, and wasted lane-vertex positions;
- distance reset, mapping writes, run-preparation writes, and their total;
- visited, active, and zero CSR/CSC runs;
- reusable-allocation status; and
- maximum concurrent workspaces under the supplied budget.

All byte arithmetic is checked before reservation. The reusable host
reservation grows each component geometrically and records growth generation;
it does not allocate device memory.

## Bounded synthetic evidence

The final clean Release integration execution used the tiny deterministic
acceptance fixture. Index, planner, and run-image values below are single host
observations. The compact-mapping value is the mean of 1,024 identical warmed
preparations so it remains observable on a coarse host clock. These values are
structural evidence only; they are not performance thresholds or real-workload
predictions.

Budget and selected configuration:

| Field | Final bounded value |
| --- | ---: |
| Plan fingerprint | `16700197086403570866` (FNV-1a over the golden plan TSV) |
| Graph vertices / tiles | `7` / `4` (serialized decision identity counts) |
| Device capacity | `65536` bytes |
| Resident graph allowance | `4096` bytes |
| Explicit reserve | `4096` bytes |
| Lane width | `8` |
| Distance slots | `2` |
| Selected-region index build | `708` ns |
| Batch planner build | `1250` ns |
| Selected mapping build | `0` ns |
| Compact bias mapping build | `11` ns (mean of 1,024 warmed builds) |
| Selected retained-run build | `583` ns |
| Descriptor-run build | `458` ns |

Four-way estimates:

| Vertex storage | Run storage | Total bytes | Preparation writes | Mapping build | Run build | Wasted lane vertices | Maximum concurrent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | Retained | `1012` | `152` | `0` ns | `583` ns | `43` | `56` |
| Full | Descriptors | `1064` | `184` | `0` ns | `458` ns | `43` | `53` |
| Compact | Retained | `1028` | `184` | `11` ns | `583` ns | `43` | `55` |
| Compact | Descriptors | `1080` | `216` | `11` ns | `458` ns | `43` | `53` |

Preparation writes are exact for the recorded warmed, identical-reuse step:
the reports count prior touched entries cleared plus current entries written.
Separate bounded tests also validate cold zero-initialization traffic. A
reusable description is bound to its first selected run representation, so a
retained-mask image cannot silently switch to descriptors or vice versa. Run
build time covers the dual-orientation host proof image (CSR and CSC); device
capacity and preparation bytes use the larger single active orientation. That
device model assumes a reused workspace stays bound to one orientation (for
example, CSC for Phase 14 Jacobi); switching CSR/CSC requires a separate
workspace or separately measured switch traffic.

For the provisionally selected full-plus-retained, two-slot strategy, the same
four-query fixture was also estimated at every required width:

| Width | Total bytes | Preparation writes | Wasted lane vertices | Maximum concurrent |
| ---: | ---: | ---: | ---: | ---: |
| 8 | `1012` | `152` | `43` | `56` |
| 16 | `1716` | `152` | `99` | `33` |
| 32 | `3124` | `152` | `211` | `18` |

The width growth here is padded lane-state and metadata, while the selected
reset traffic stays fixed because the fixture still has four valid queries.

The bounded fixture independently proves that retained masks and compact
descriptors encode the same nonzero CSR/CSC run masks and the same admitted
lane/edge pairs. It also exercises touched-ledger reuse and validates every
reported component before accepting a serialized decision. The decision binds
the plan fingerprint, graph vertex/tile counts, the complete 2x2 strategy
matrix, byte/preparation formulas, budget-derived concurrency, and selected
timings. Inconsistent or coordinated row-only mutations that disagree with the
recorded dimensions, plus serializer control characters, are rejected. These
recorded counts are integrity inputs, not a cryptographic graph signature.

The provisional full-plus-retained choice follows the Phase 13 rule to prefer
the simpler full layout when capacity is safe and reset/scanning work is
acceptable. On the acceptance fixture it:

- needs no global/local distance-label mapping;
- preserves direct graph-vertex indexing for all future engines;
- reuses one retained mask allocation and clears only previously touched runs;
  and
- exposes a simpler future kernel view than mapped labels plus descriptor
  offsets.

The compact alternatives save label or run storage on the fixture. Those
savings are not promoted because a tiny graph cannot establish real union
sparsity, mapping cost, run sparsity, or GPU traversal cost.

## Recorded Phase 7 analytical label bytes

The partial Phase 7 report records:

```text
vertices = 28,226,432
directed edges = 130,278,682
graph artifact bytes = 3,958,293,872
real query artifact = not emitted
```

Only the vertex count is used below. It gives exact distance-label bytes from
`V * lane_width * distance_slots * sizeof(float)`:

| Width | One slot | Two slots | Two slots (GiB) |
| ---: | ---: | ---: | ---: |
| 8 | 903,245,824 | 1,806,491,648 | 1.682426 |
| 16 | 1,806,491,648 | 3,612,983,296 | 3.364853 |
| 32 | 3,612,983,296 | 7,225,966,592 | 6.729706 |

These are V-only label-array values. They exclude the resident device graph,
tile mapping, run masks/descriptors, descriptor offsets, controllers,
terminals, queues, counters, allocator overhead, and other concurrent work.
The graph artifact file size is not substituted for a resident-device graph
report.

For one illustrative analytical scenario only, assume:

```text
nominal capacity = 64 GiB = 68,719,476,736 bytes
resident graph allowance = 8 GiB = 8,589,934,592 bytes
explicit reserve = 4 GiB = 4,294,967,296 bytes
distance slots = 2
```

The width-32 two-slot label component is arithmetically smaller than the
remaining illustrative envelope. This is not measured free memory, not an AUP
allocation result, not a maximum-concurrency result, and not Phase 13
acceptance evidence.

## Why production selection is deferred

The interrupted Phase 7 run created and round-tripped the graph artifact but
did not emit:

```text
logicnets_jscl.padding1.v1.bfqueries
vtr_mcml_unrouted.padding1.v1.bfqueries
all_inputs.v1.tsv
the complete phase7_report.v1.txt
```

Without the real query corpus, Phase 13 cannot truthfully measure:

- batch occupancy and singleton rates at widths 8, 16, and 32;
- union-tile inflation and selected-versus-wasted vertices;
- active/zero CSR and CSC run ratios;
- retained clear/write traffic versus descriptor emission;
- compact global/local mapping build time;
- runtime free memory after the resident graph upload;
- maximum concurrent real workspaces; or
- preparation and run traversal on `gfx1151`.

The provisional choice may become production only after all of those inputs
are captured with workload identity and repeated timing evidence.

## Validation and deferred commands

The bounded CPU commands below were run for Phase 13. The real-artifact and
eventual device commands remain instructions only and were not run.

### Bounded CPU acceptance (completed)

This remains safe on a CPU-only host and performs no real-graph search:

```bash
cmake -S . -B build/phase13-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBFNEW_ENABLE_HIP=OFF \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON
cmake --build build/phase13-cpu --parallel
ctest --test-dir build/phase13-cpu --output-on-failure
ctest --test-dir build/phase13-cpu --output-on-failure \
  -R '^(bfnew\.batch_plan|bfnew\.batch_workspace)$'
```

The recorded result is `14/14` Release tests in `0.05` seconds. A separate
focused invocation passed `2/2` in `0.26` seconds, and the focused ASan+UBSan
build passed `2/2` in `0.45` seconds.

### Complete the real query artifacts after implementation phases (deferred)

This is a long full-scale CPU preprocessing run. Do not run it as a routine
Phase 13 local test. When the maintainer schedules the combined campaign, use a
fresh output directory so the Phase 7 partial record remains untouched:

```bash
cmake -S . -B build/phase13-fpgaif \
  -DCMAKE_BUILD_TYPE=Release \
  -DBFNEW_ENABLE_HIP=OFF \
  -DBFNEW_ENABLE_FPGAIF=ON
cmake --build build/phase13-fpgaif \
  --target bfnew_build_fpga_workload --parallel
mkdir -p build/phase13-fpgaif/workload-logs out/phase13-workload
/usr/bin/time -p \
  -o build/phase13-fpgaif/workload-logs/wall.txt \
  build/phase13-fpgaif/bfnew_build_fpga_workload \
    --data-root /Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest \
    --output-root out/phase13-workload \
    --medium vtr_mcml_unrouted.phys \
    --tile-width 8 \
    --tile-height 8 \
    --padding 1 \
  >build/phase13-fpgaif/workload-logs/run.log 2>&1
```

Require all artifact round-trip, deterministic-byte, query-validation,
bounded-Dijkstra-sample, and input-unchanged checks to pass. A partial output
does not promote the workspace decision.

### Eventual device validation (deferred)

Phase 13 has no batched GPU executable. Phases 14 and 15 now provide separate
HIP source paths, but neither has been compiled or executed on a GPU. After all
implementation phases are complete, first complete the generic HIP gate and
single-query campaign in `docs/PHASE12_GPU_VALIDATION.md`. Then record runtime
device capacity/free bytes after resident graph upload, prepare both run
representations for the identical fingerprinted plans, and separate mapping,
reset, run-build, kernel, and result-transfer intervals. Profiler-instrumented
runs must remain separate from ordinary timing.

Do not use a browser, `sjc.aupcloud`, or a local non-HIP machine for any of
those device runs.

## Phase 15 consumption of the provisional choice

Phase 15 consumes the full graph-sized, vertex-major label layout with the
retained CSR run-mask image. Dense push needs one unsigned 32-bit distance word
per vertex/lane rather than Phase 14's two float slots. The reusable Phase 15
workspace therefore retains one full-vertex lane array and initializes only
selected range/lane words for each prepared batch. Its source-seed count is the
zero-valued subset of those selected one-slot writes. Total reserved capacity
and convenient portable initialization of nonselected cells are not reset
traffic.

The portable Phase 15 model also proves retained-mask/compact-descriptor
semantic parity. That proof does not implement compact descriptors in the HIP
engine and does not promote them to the production choice. Likewise, its
`csr_edge_loads` counter records logical shared edge-record requests from the
algorithm; it is not physical cache traffic and cannot decide between
workspace or load strategies.

This remains a provisional implementation choice. The real query artifact,
real union and CSR-run distributions, runtime free memory, device allocation,
device correctness, physical traffic, and timing are still unavailable. The
combined campaign must measure them before any production recommendation.

## Phase 16 consumption and additional frontier state

Phase 16 retains the provisional full graph-sized, vertex-major one-slot label
layout and exact retained CSR run-mask image. Its distance component is the
same `vertex_count * lane_width * sizeof(uint32_t)` term modeled by the Phase 13
one-slot rows. The active-frontier engine also requires two bounded 32-bit
vertex queues and two graph-sized 32-bit per-vertex lane-mask arrays.

Those queues and activity masks were deliberately outside the generic Phase 13
distance/run comparison. With graph vertex count `V`, queue capacity `Q`, and
lane width `W`, the Phase 16 hot scratch is:

```text
distance words:        4 * V * W
two vertex queues:     8 * Q
two activity masks:    8 * V
total:                 4 * V * W + 8 * Q + 8 * V bytes
```

The default `Q = V` simplifies this to `4 * V * W + 16 * V`. At the recorded
Phase 7 vertex count `V = 28,226,432`, the exact default-capacity scratch bytes
are:

| Width | Distance bytes | Queue + mask bytes | Total scratch bytes |
| ---: | ---: | ---: | ---: |
| 1 | 112,905,728 | 451,622,912 | 564,528,640 |
| 8 | 903,245,824 | 451,622,912 | 1,354,868,736 |
| 16 | 1,806,491,648 | 451,622,912 | 2,258,114,560 |
| 32 | 3,612,983,296 | 451,622,912 | 4,064,606,208 |

The checked public `make_batched_frontier_scratch_layout` and deferred
`hip::batched_frontier_scratch_bytes` APIs use this same order and formula:
distance words, two activity-mask arrays, then two vertex queues. The bounded
deferred test source checks their exact equality at widths 1/8/16/32 for both
default `Q = V` and explicit `Q = 1`.

These are checked arithmetic values, not allocations, observed free memory,
or evidence that a complete resident graph plus workspace fits. They still
exclude terminal/range metadata, controller/status/statistics, run masks,
allocator overhead, and other concurrent work. A smaller explicit queue
capacity exists only for overflow validation and is not a production memory
selection.

The portable Phase 16 path again proves retained-mask/compact-descriptor
semantic parity, while the first HIP path consumes retained CSR masks only.
That equivalence does not implement compact device storage or promote the
bounded-synthetic choice to production. Real union sizes, frontier high-water
marks, runtime free memory, device allocations, and measured traffic/timing
remain required before a production workspace decision.

## Phase boundary

Phase 13 stops at deterministic plans, reusable batch descriptions, exact run
admission, checked estimates, and this provisional decision. It contains no
batched SSSP kernel. Phase 14 consumes the provisional full-vertex/retained-
CSC two-slot form for overlapping batched Jacobi pull. Phases 15 and 16 consume
the provisional full-vertex/retained-CSR one-slot form for dense and active-
frontier push respectively, with Phase 16 accounting separately for queues and
activity masks. Phase 17 now reuses those engine workspaces sequentially across
expansion generations and adds no separate device expansion allocation. Its
real capacity and retry evidence remain deferred.
