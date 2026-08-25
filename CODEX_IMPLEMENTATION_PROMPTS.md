# Clean-Room Weighted Bellman-Ford Prototype: Codex Prompt Pack

> **Campaign status:** Phases 0–5 in this document are the accepted historical
> foundation. The Phase 6 prompt and everything after it are superseded by
> `GPU_SSSP_CONTINUATION_PROMPT_PACK.md` and must not be executed. The project
> contract contains the current inspection-and-independent-adaptation policy.

This document is a sequence of small prompts for building a new weighted,
nonnegative-edge SSSP prototype in this directory. Use one phase prompt at a
time. Do not paste all phases into a coding agent at once.

At the start of a new coding-agent task, provide the **Global contract** and
then exactly one **Phase prompt**. Continue to the next phase only after the
current phase's tests and acceptance checks pass.

## Global contract (paste before every phase prompt)

You are implementing one small phase of a clean-room weighted shortest-path
prototype.

Workspace and ownership rules:

- The only writable project directory is
  `/Users/amber_bajaj/Desktop/RIPS/bf-new`.
- Treat `/Users/amber_bajaj/Desktop/RIPS/rips2026-amd-routing` as read-only
  historical background. Do not copy, transcribe, adapt, import, link, or
  derive code, tests, names, serialized formats, constants, or implementation
  details from it. Do not add it to the build. Implement everything in
  `bf-new` from first principles.
- Treat `/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest` as read-only
  input data and an external schema/API source. It may be read and linked as a
  dependency, but never edited. Generate Cap'n Proto outputs under
  `bf-new/build`, never beside the source schemas.
- Put every generated report, cache, converted graph, and temporary artifact
  under `bf-new/build`, `bf-new/out`, or an OS temporary directory.
- Before finishing, verify that neither read-only repository was modified.

Algorithm rules:

- This project supports arbitrary finite **nonnegative** edge weights. There
  are no negative edges or negative cycles.
- Do not implement BFS, bitset BFS, unit-weight special cases, or code paths
  that assume every weight is one. Tests must include fractional and unequal
  positive weights, zero-weight edges, parallel edges, and disconnected
  vertices.
- Dijkstra may exist only as a CPU correctness oracle. The prototype under
  test is a weighted min-plus/Bellman-Ford family implementation.
- Do not store predecessors in the hot relaxation workspace. Reconstruct
  requested paths afterward from final distances and incoming CSC edges.
- All results must be checked against the Dijkstra oracle on CPU before a
  phase is considered complete.
- Determinism is required for graph construction, ordering, batching, reports,
  and tests.

Performance and memory rules:

- Profiling has shown approximately 8 GB used out of 64 GB available GPU
  memory. Treat device-memory **capacity** as ample: duplicate performance-
  critical CSR/CSC fields when doing so avoids indirection.
- This capacity result does not prove that memory bandwidth, latency, cache
  behavior, or atomic contention are irrelevant. Keep counters for examined
  edges, useful relaxations, frontier sizes, push/pull work estimates, and
  batch-lane utilization. Do not claim a bottleneck without measurements.
- Prefer structure-of-arrays storage, contiguous lane values, fixed-width
  IDs, stable edge IDs, and trivially copyable device-facing views.
- HIP/ROCm is unavailable on this computer. All tests in the current campaign
  must compile and run on CPU. Do not claim HIP code works without testing it
  later on a ROCm system.

Process rules:

- Read `CODEX_IMPLEMENTATION_PROMPTS.md`, `PROJECT_CONTRACT.md` if present,
  `docs/DESIGN.md`, and the current implementation ledger before editing.
- Inspect the current tree and preserve completed behavior.
- Implement only the requested phase. Do not start later phases.
- Keep each source file focused; split a file before it becomes a catch-all.
- Use C++20, CMake, CTest, and the standard library. Do not add a network
  dependency merely for testing.
- Run the phase's focused tests, then the complete CPU test suite.
- Update `docs/IMPLEMENTED_OPTIMIZATIONS.md` with only functionality that now
  exists and has passing evidence. Clearly label planned work as not
  implemented.
- End with: files changed, commands run, test results, measured findings,
  remaining limitations, and the exact next phase number. Stop after that.

## Target project layout

The exact split may evolve, but keep approximately this architecture. The CPU
campaign should finish with about 30 C++ files: 8 public headers, 11 library
sources, 3 tools, and 8 focused tests. Build and documentation files are in
addition to those. HIP files are a later, separately tested campaign.

```text
bf-new/
  CMakeLists.txt
  cmake/
    InterchangeSchemas.cmake
  include/bfnew/
    types.hpp
    graph.hpp
    spatial.hpp
    query.hpp
    batch.hpp
    sssp.hpp
    stats.hpp
    interchange.hpp
  src/
    graph_builder.cpp
    reorder.cpp
    tiled_graph.cpp
    dynamic_region.cpp
    batch_planner.cpp
    path_reconstruct.cpp
    cpu/reference_dijkstra.cpp
    cpu/frontier_sssp.cpp
    cpu/batched_spmm.cpp
    io/interchange_reader.cpp
    io/binary_graph.cpp
  tools/
    analyze_phys.cpp
    build_device_graph.cpp
    run_cpu_prototype.cpp
  tests/
    graph_test.cpp
    reorder_test.cpp
    tile_test.cpp
    sssp_test.cpp
    expansion_test.cpp
    batch_overlap_test.cpp
    batch_disjoint_test.cpp
    interchange_smoke_test.cpp
  docs/
    DESIGN.md
    CPU_RESULTS.md
    HIP_PORT.md
    IMPLEMENTED_OPTIMIZATIONS.md
  out/
```

Do not create empty placeholder source files far ahead of the phase that needs
them. The tree is a destination, not a requirement to add everything in Phase
0.

## Required in-memory structures

Use strong integer types or wrappers for `VertexId`, `EdgeId`, `TileId`, and
`QueryId`. Prefer 32-bit vertex/tile/query IDs and 64-bit edge counts/offsets.
Validate all conversions.

### Immutable dual graph

```text
WeightedGraph
  vertex_count: uint32/size_t
  edge_count: uint64/size_t

  outgoing CSR
    row_offsets[n + 1] : uint64
    destinations[m]    : VertexId
    weights[m]         : float initially, nonnegative and finite
    edge_ids[m]        : EdgeId

  incoming CSC
    column_offsets[n + 1] : uint64
    sources[m]            : VertexId
    weights[m]            : duplicated for contiguous pull access
    edge_ids[m]           : the same stable logical IDs as CSR

  vertex metadata (structure of arrays)
    x[n], y[n]                 : int32, with explicit missing flag
    resource_class[n]          : compact integer/interned class
    owner_tile[n]              : TileId
    original_vertex_id[n]      : VertexId

  permutations
    old_to_new[n]
    new_to_old[n]
```

CSR rows must be ordered by destination tile, then destination ID, then stable
edge ID. CSC columns must be ordered by source tile, then source ID, then
stable edge ID. Duplicate weights across CSR and CSC deliberately; the project
has memory-capacity headroom, and pull should not need a random edge-weight
indirection.

### Spatial tiling and halos

Implement a uniform grid first behind a `SpatialPartitioner` interface. A
quadtree may be evaluated later from measured tile distributions; do not build
both initially.

```text
TileDirectory
  grid parameters and coordinate bounds
  tile_vertex_offsets[T + 1]  # reordered vertices are contiguous by tile
  neighbor_tile_offsets/lists
  internal edge ranges
  outgoing cross-tile edge ranges
  incoming cross-tile edge ranges
  halo vertex ranges/lists
  special unlocated/spill tile
```

A halo is metadata about a remote endpoint adjacent to an owned vertex. It
does not own a second distance value. Global state has one value per reordered
vertex. Vertices without valid coordinates must go into an explicit special
tile with documented admission behavior; never silently drop them.

### Queries, regions, and batches

```text
RouteQuery
  query_id
  source vertices
  target vertices
  selected tile bitset/list
  expansion state

BatchPlan
  mode: DisjointRegions | OverlapLanes
  query IDs
  union tile list
  per-tile active-query lane mask
  padded lane count

OverlapWorkspace (vertex-major, lanes contiguous)
  distances[selected_vertex_count * lane_count]
  active_lane_mask[selected_vertex_count]
  next_lane_mask[selected_vertex_count]
  changed/touched metadata

DisjointWorkspace
  fused entries carrying query ownership, or a proved single-owner region map
  one scalar distance per admitted (query, vertex) state
```

For overlap batches, use at most 64 lanes in the first CPU prototype so an
active-query set fits in `uint64_t`. A lane is active on an edge only when the
source and destination tiles are selected for that query.

### Exact dynamic-region certificate

For a query solved inside selected tiles, track every edge `(u, v)` with `u`
inside and `v` outside. With nonnegative weights, define:

```text
exit_lower_bound = min(distance_inside[u] + weight(u, v))
```

The bounded answer is globally exact if every target is reached and every
target distance is at most `exit_lower_bound`. Otherwise expand by one tile
ring and continue/restart according to the tested policy. Include a test where
a target is reachable inside the initial region but a cheaper path exits and
re-enters; reachability alone must fail while the certificate triggers
expansion.

## Phase prompts

### Phase 0 — establish the clean-room contract and scaffold

Create only the project contract, minimal CMake/CTest scaffold, documentation
skeleton, and one trivial build test. Detect C++20 support. Do not implement a
graph or algorithm yet.

Write `PROJECT_CONTRACT.md` from the global contract in this prompt. Record the
absolute read-only FPGA24 data/schema roots as configurable CMake cache paths,
not hard-coded library dependencies. Add compiler warnings. Tests must require
no downloaded framework.

Acceptance:

- A fresh `cmake -S . -B build` and `cmake --build build` succeeds.
- `ctest --test-dir build --output-on-failure` passes.
- No file outside `bf-new` changes.
- The implementation ledger says no algorithmic optimization exists yet.

### Phase 1 — core weighted graph types and synthetic fixtures

Implement the strong IDs, edge input record, vertex metadata, an initially
simple graph interface, validation helpers, and deterministic synthetic graph
fixtures. Do not build CSR/CSC yet.

Fixtures must cover unequal fractional weights, a zero-weight edge, parallel
edges, a self-loop, a disconnected vertex, multiple targets, missing spatial
coordinates, and at least two resource classes. Reject NaN, infinity, negative
weights, and out-of-range IDs.

Acceptance: focused graph/type tests plus the complete suite pass. No BFS or
unit-weight optimization appears anywhere.

### Phase 2 — deterministic dual CSR/CSC construction

Build immutable outgoing CSR and incoming CSC from edge records. Assign stable
logical edge IDs before layout sorting. Duplicate weights in both views. Add a
deep validator proving that CSR and CSC contain the same multiset of logical
edges and weights, offsets are monotonic, and every row/column is ordered
deterministically.

Do not add spatial reordering yet. Test empty rows, duplicate endpoints,
parallel edges, self-loops, and graphs whose edge count exceeds a 32-bit
offset assumption using simulated boundary checks if a huge allocation is not
reasonable.

Acceptance: transpose/multiset invariants and deterministic rebuild tests pass.

### Phase 3 — spatial vertex/edge ordering

Implement deterministic vertex permutation by:

1. spatial tile key,
2. resource class,
3. Morton key derived from available representative/end-point coordinates,
4. original vertex ID as the final tie-breaker.

If multiple vertices have only one identical tile coordinate and no meaningful
within-tile coordinate, document that fact and use resource class plus original
ID rather than inventing geometry. Keep the ordering policy injectable so a
Hilbert policy can be tested later.

After vertex permutation, rebuild CSR and CSC. Order each outgoing row by
destination tile to improve destination-label locality, and each incoming
column by source tile. Verify permutation round trips, edge preservation,
determinism, and contiguous tile spans.

Acceptance: reorder tests pass and print a small before/after locality report
for a synthetic spatial graph.

### Phase 4 — uniform tiles, cross-tile edges, and halos

Implement `SpatialPartitioner` and the uniform-grid version only. Construct the
tile directory, neighboring-tile lists, internal edges, incoming/outgoing
cross-tile edge metadata, and halo vertex lists without duplicating distance
state. Implement the special unlocated/spill tile explicitly.

Provide validation for ownership uniqueness, halo validity, edge accounting,
and symmetric neighbor metadata where appropriate. Test long edges that skip
tiles as well as adjacent-tile edges.

Acceptance: every graph edge is classified exactly once for its source owner,
all halo endpoints resolve to a global reordered vertex, and tile tests pass.

### Phase 5 — weighted CPU oracles and post-relaxation path reconstruction

Implement two CPU correctness algorithms:

- Dijkstra for nonnegative weights, used only as the oracle.
- A simple synchronous min-plus Bellman-Ford reference using arbitrary
  nonnegative weights.

Do not implement BFS or branch on unit weights. Store distances only during
relaxation. Reconstruct requested paths afterward using incoming CSC edges and
the final distance labels. Define deterministic tie-breaking by stable edge ID.
Use exact comparisons for deliberately representable test values; document
the floating-point comparison policy for general data.

Acceptance: both algorithms agree on all fixture distances and reconstructed
paths are valid, have the reported cost, and terminate at a source.

### Phase 6 — edge-balanced push, CSC pull, and adaptive switching on CPU

Implement the production CPU semantic prototype:

- active-frontier weighted min-plus iterations,
- edge-balanced push by prefix-summing active-row degrees and flattening active
  edges into deterministic chunks,
- destination-owned pull over CSC,
- an injectable push/pull selector using estimated edge work,
- convergence and target completion entirely within the algorithm call.

Pull must compute the minimum over all relevant incoming candidates; do not use
Boolean/BFS early-exit logic. Keep instrumentation for iterations, frontier
vertices, active edges, examined edges, successful improvements, push rounds,
pull rounds, and estimated work.

Test forced-push, forced-pull, and adaptive modes against Dijkstra on random
nonnegative weighted graphs and the deterministic fixtures.

Acceptance: all modes return identical distances. The output includes counters
but makes no unsupported performance claim.

### Phase 7 — exact dynamically expanding tiled regions

Add query regions initialized from source and target tiles plus configurable
tile padding. Restrict relaxation to the induced selected-tile subgraph. Track
exit edges and compute the nonnegative exit lower-bound certificate described
above. Expand by one neighboring-tile ring when targets are unreachable or the
certificate cannot prove exactness.

Handle the special unlocated/spill tile according to the policy documented in
Phase 4. Avoid an immediate whole-graph fallback. Record initial/final tile
counts, expansions, examined boundary edges, and certificate values.

Acceptance includes:

- a path entirely inside the initial region,
- an unreachable initial region that expands,
- a reachable but suboptimal inside path whose cheaper global path exits and
  re-enters,
- a long cross-tile edge,
- equality with full-graph Dijkstra for every case.

### Phase 8 — overlapping/similar-query batched sparse min-plus SpMM

Implement the first batching mode, `OverlapLanes`, for 1–64 weighted queries.
Use vertex-major distances with contiguous lanes and one `uint64_t` active-lane
mask per vertex. A graph edge is loaded once by the CPU semantic kernel and
applied to all active lanes admitted by their per-query tile masks.

Support batched push and batched pull semantics. Do not merge sources across
lanes. Track useful lane operations, padded/inactive lane operations, edge
loads, per-round lane occupancy, and union-tile waste. Reconstruct only
requested target paths after distances converge.

Acceptance: every lane agrees with running the single-query weighted solver
independently, including queries with overlapping sources, targets, and tiles.

### Phase 9 — disjoint-region batching

Implement `DisjointRegions` as a separate batch plan and execution path.
Prove disjointness from selected vertex/tile regions and admitted cross edges,
not merely non-overlapping terminal rectangles. A fused pseudo-source or
single-owner scalar layout is allowed only after this proof. Preserve query
identity for results and path reconstruction.

If dynamic expansion would invalidate disjointness, deterministically split or
replan the batch; never silently mix distances from different sources. Compare
all results with independent Dijkstra and with the overlap-lane executor.

Acceptance: safe disjoint cases fuse, cross-edge/halo counterexamples are
rejected or replanned, and all distances remain exact.

### Phase 10 — deterministic batch planner

Implement batch planning separately from execution. Inputs are query tile sets,
terminal boxes, estimated selected edges, and a maximum lane count. Produce:

- disjoint batches when the strong region/edge proof succeeds,
- overlap-lane batches using tile-set Jaccard similarity and bounded union
  waste,
- deterministic leftovers for singleton execution.

Make scoring weights configurable. Do not hard-code a batch width as optimal.
Emit planner metrics: pairwise overlap distribution, batch-size histogram,
tile-union inflation, estimated edge reuse, and rejected-disjoint reasons.

Acceptance: deterministic golden tests cover clustering, tie-breaking, maximum
lanes, and unsafe disjointness.

### Phase 11 — read-only `.phys` benchmark analyzer

Add C++ Cap'n Proto and zlib support. Use the schemas from:

`/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest/fpga-interchange-schema/interchange`

Generate C++ schema sources into `bf-new/build/generated`. Read gzip-compressed
FPGA Interchange files without creating decompressed copies beside the inputs.

Build `analyze_phys` to scan `.phys` files without constructing the full RRG.
Report net counts, source/stub counts, fanout, available site/tile coordinate
coverage, terminal rectangles, spatial extent, rectangle overlap, candidate
batch-size distributions, and missing-coordinate rates. Make no claim that a
`.phys` file contains the full device routing graph; it does not.

Start with `logicnets_jscl_unrouted.phys`, then smoke-test at least one medium
benchmark. Write JSON/CSV results under `bf-new/out`. Do not change input
timestamps or files.

Acceptance: repeat runs are byte-for-byte deterministic and malformed/missing
input errors are clear.

### Phase 12 — read-only `.device` metadata scan

Extend the interchange reader to scan `xcvu3p.device` without building the
full graph. Report tile coordinate bounds, tile-type/resource distributions,
node/wire/PIP counts, representative-coordinate coverage, potential resource
classes, degree estimates that can be computed safely, and memory estimates
for dual CSR/CSC plus metadata.

Do not solve terminal mapping or build all edges in this phase. Add command
limits/sampling only if they are explicit and reported; never present a sample
as a full-device result.

Acceptance: the scan completes or fails with a measured resource explanation,
outputs deterministic JSON, and changes nothing outside `bf-new`.

### Phase 13 — construct bounded and full device routing graphs

Using the FPGA Interchange schema as the specification, implement clean-room
conversion from DeviceResources nodes/wires/PIPs to the Phase 2 edge records.
Start with a configurable coordinate rectangle, validate it deeply, then make
full-device construction an explicit opt-in command. Respect PIP direction and
record stable provenance sufficient to diagnose an edge without retaining
large strings in hot arrays.

Apply the measured resource-class mapping, spatial reorder, dual CSR/CSC build,
and tiling. Serialize the resulting host graph to a documented, versioned
`bf-new/out` format for repeatable CPU experiments. Do not use or imitate a
format from the historical repository.

Acceptance: bounded-graph counts and sampled edges are independently checked
against schema records; serialize/load round trips pass; memory and timing are
reported honestly.

### Phase 14 — map physical-net terminals to graph queries

Implement source/stub site-pin-to-routing-node mapping from the FPGA
Interchange schemas and documented device relationships. Keep mapping logic
separate from graph construction. Report ambiguous, absent, and multi-source
cases explicitly. Never guess silently.

Create `RouteQuery` records for a small benchmark subset and validate terminal
coordinates, graph vertex ranges, and reachability sampling. This phase does
not write routed `.phys` files.

Acceptance: hand-inspected samples show the original net/site pin, mapped
vertex, coordinate, and resource class; failure categories are counted; all
mapped queries pass structural validation.

### Phase 15 — integrated CPU experiments and parameter recommendations

Run the complete weighted CPU prototype on representative sampled queries from
at least a small and a medium benchmark. Compare:

- full-graph reference versus dynamically tiled execution,
- forced push, forced pull, and adaptive switching,
- singleton versus overlap-lane batches,
- singleton versus proved disjoint batches,
- several tile sizes, initial padding values, and batch widths.

Every sampled result must agree with Dijkstra. Report P50/P95/P99 query time,
examined edges, useful relaxations, expansions, selected tile/edge counts,
lane utilization, estimated edge reuse, and graph/workspace memory. Separate
parsing/build time from repeated SSSP time.

Write `docs/CPU_RESULTS.md` and recommend values only from measured data. Do not
infer GPU speedups from CPU runtime.

Acceptance: commands and seeds are recorded, reports are reproducible, and
all correctness checks pass.

### Phase 16 — CPU campaign audit and ROCm handoff

Do not add new algorithms. Audit the code, tests, reports, and implementation
ledger. Remove stale claims and ensure each claimed optimization points to
specific source files, tests, and measured counters. Document the POD/device
views, ownership, buffer sizes, synchronization semantics, and kernel inputs
needed for a later HIP port in `docs/HIP_PORT.md`.

Produce a final summary with four categories:

1. implemented and CPU-tested,
2. implemented but not performance-validated,
3. designed but not implemented,
4. ROCm-only work that cannot be validated on this machine.

Acceptance: no optimization is described as implemented without source and
passing-test evidence.

## Later ROCm-system prompts — do not execute on the current computer

These phases belong in a new campaign on a machine with supported HIP/ROCm.
Use the same global contract, but require real GPU compilation, sanitizer or
debug validation where available, CPU/GPU distance comparison, and profiler
evidence at every phase.

### ROCm Phase A — device views and transfer validation

Implement trivially copyable device views of the dual graph, tile directory,
queries, and batch plans. Add HIP allocation/upload/download RAII and verify
byte-for-byte metadata plus sampled arrays on the GPU. No SSSP kernel yet.

### ROCm Phase B — single-query edge-balanced push

Implement weighted distance-only push with flattened active edges, 32-bit
distance updates where valid, queue compaction, and GPU-resident convergence.
Compare every result with the CPU oracle. Profile edge balance, atomics,
occupancy, cache behavior, and end-to-end time.

### ROCm Phase C — destination-owned CSC pull and adaptive switching

Implement weighted pull, one logical owner per destination, and a measured
push/pull selector. Validate exact equality with CPU results and profile the
switching regimes. Do not use BFS early exit.

### ROCm Phase D — overlap-lane sparse min-plus SpMM

Implement 8/16/32/64-lane weighted batches, per-query tile masks, and
vertex-major contiguous lane storage. Measure edge reuse, lane utilization,
register pressure, occupancy, cache traffic, and total query throughput.

### ROCm Phase E — disjoint batches, dynamic expansion, and final audit

Implement proved disjoint batches, dynamic expansion/replanning, GPU-side
certificate reduction, and post-relaxation path reconstruction. Compare all
targets and sampled paths with the CPU prototype. Update the implementation
ledger with profiler-supported conclusions.

## Intended optimization ledger

The following are requirements, not claims about the current empty directory.
Each item becomes "implemented" only after its phase passes:

- dual outgoing CSR and incoming CSC,
- duplicated contiguous pull weights,
- spatial vertex ordering by tile/resource/locality key,
- outgoing edges grouped by destination tile,
- distance-only hot relaxation and post-pass path reconstruction,
- uniform pre-tiling with cross-tile metadata and halos,
- exact nonnegative dynamic-region expansion certificate,
- edge-balanced active-edge push,
- destination-owned weighted CSC pull,
- measured adaptive push/pull switching,
- overlapping/similar-query batched sparse min-plus SpMM,
- strongly proved disjoint-region batching,
- deterministic spatial batch planning,
- read-only FPGA24 `.phys` and `.device` analysis,
- CPU correctness validation against nonnegative-weight Dijkstra,
- HIP-ready POD memory views and an explicitly deferred ROCm campaign.

At the time this prompt pack was created, `bf-new` contained no implementation,
so none of these optimizations had yet been implemented.
