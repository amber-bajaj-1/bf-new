# Implemented Optimizations

## Implemented and CPU-tested through Phase 5

Strong fixed-width IDs, physical provenance, vertex metadata, validated
weighted directed edge records, an immutable record-level input graph, and a
deterministic synthetic fixture are implemented and covered by
`tests/graph_test.cpp`. Passing evidence is the `bfnew.graph` CTest. These are
correctness foundations, not algorithmic optimizations.

Phase 2 now implements:

- immutable outgoing CSR and incoming CSC with 64-bit offsets;
- canonical stable logical edge IDs independent of sparse-array positions and
  incidental input-record order;
- duplicated contiguous float weights for outgoing and incoming traversal;
- deterministic destination/edge-ID row order and source/edge-ID column order;
  and
- deep offset, ordering, logical-multiset, exact-weight, and transpose
  validation.

Passing evidence is the `bfnew.graph` CTest, including deterministic rebuilds,
empty rows and graphs, parallel and duplicate records, a self-loop, and a
simulated edge offset above 32 bits. These structural optimizations are not yet
performance-validated, and no shortest-path algorithm exists.

Phase 3 now implements:

- checked uniform-grid coordinate derivation for preprocessing;
- dense row-major tile IDs plus an explicit spill tile;
- deterministic vertex permutation by tile, resource class, Morton locality,
  and original ID;
- identity-preserving old/new permutations and contiguous per-tile vertex
  spans;
- outgoing edges grouped by destination tile; and
- incoming edges grouped by source tile.

Passing evidence is the `bfnew.reorder` CTest. It covers permutation round
trips, row-major and negative tile coordinates, identical-coordinate fallback,
spill ordering, injectable locality keys, edge preservation, deep sparse-view
validation, and byte-identical repeat construction. The synthetic locality
report changes destination-tile runs from 7 to 4; no runtime benefit is claimed.

Phase 4 now implements:

- a preprocessing-only `SpatialPartitioner` interface and uniform-grid
  implementation;
- deterministic geometric 8-neighbor lists for located tiles;
- actual-edge-only spill adjacency;
- source-owned internal and outgoing cross-edge ranges;
- destination-owned incoming cross-edge ranges;
- deduplicated remote-endpoint halo lists without distance duplication; and
- deep ownership, edge-accounting, neighbor, and halo validation.

Passing evidence is the `bfnew.tile` CTest. It covers adjacent and diagonal
neighbors, a long tile-skipping edge, explicit spill behavior, exact source
classification, incoming cross completeness, globally resolvable halos, and
byte-identical repeat construction. No runtime benefit is claimed.

Phase 5 now implements correctness references rather than an optimization:

- CPU Dijkstra for a nonnegative-weight oracle;
- synchronous weighted min-plus Bellman-Ford with float candidates;
- distance-only relaxation state in both algorithms;
- deterministic incoming-CSC path reconstruction after relaxation;
- stable-edge-ID tie-breaking with zero-cycle detection and backtracking; and
- exact and ULP-aware verification utilities with tolerance excluded from
  algorithm decisions.

Passing evidence is the `bfnew.sssp` CTest. It requires exact oracle/reference
agreement on representable fixtures, validates all reconstructed paths and
costs, covers multiple sources and spatially reordered graphs, and exercises a
zero-weight cycle that requires backtracking. This phase provides correctness
baselines only; no algorithmic optimization or performance claim exists yet.

## Phase 6 passed under the CPU-only acceptance policy

Phase 6 adds campaign infrastructure rather than SSSP:

- optional HIP language/compiler detection gated by `BFNEW_ENABLE_HIP`, with
  the CPU-only build remaining independent of ROCm and a clear error for an
  unavailable explicitly requested HIP toolchain;
- a GPU probe that reports target, wave width, WGP/multiprocessor and physical
  CU counts, cooperative capability plus an executed grid barrier, and
  probe-kernel occupancy;
- a nontrivial repeated arithmetic/memory kernel for real PMC collection;
- recursive compatibility checking for the required candidate counter groups,
  profiler version/catalog capture, one nonempty counter-collection gate, and
  dependency tracing; and
- a cooperative barrier matrix covering legal 128/256/512-thread blocks,
  1/2/4/kernel-maximum blocks per WGP, and multiple grid-barrier counts.

Passing evidence is the unchanged five-test CPU suite in
`build/phase6-cpu`, including exact Phase 5 Dijkstra/Bellman-Ford comparisons.
The optional HIP tools have not been compiled or executed on `gfx1151`, so no
GPU correctness, profiler, occupancy, or performance result is claimed. The
barrier benchmark is not an SSSP performance claim.

## Phase 7 passed under the CPU-only acceptance policy

Phase 7 adds host workload infrastructure rather than a GPU optimization:

- an optional, independently implemented FPGA Interchange device/physical-net
  bridge with no historical source dependency;
- directed PIP orientation, stable numeric provenance, schema-delay weights,
  and a deterministic weighted fallback for absent timing data;
- primary and alternate site-pin mapping with explicit unmappable-resource
  counters;
- ordinary-net filtering, one/two-source support, complete sink retention, and
  deterministic query IDs;
- a deeply validated `RouteQuery`, padded region selection, size estimation,
  and bounded induced-graph construction;
- explicit little-endian versioned graph/query artifacts with deep readers and
  byte-determinism checks; and
- a reporting tool for the complete logicnets corpus, a medium benchmark, all
  13 metadata scans, overlap/Jaccard and occupancy estimates, endpoint samples,
  and bounded CPU Dijkstra evidence.

Passing evidence is 7/7 CTests in `build/phase7-fpgaif`, including an
independently generated compressed FPGA Interchange fixture, unequal timing
weights, bidirectional orientation/provenance, physical-net exclusions,
artifact round trips, deep graph/tile/query validation, and bounded Dijkstra.
The default dependency-minimal build passes 6/6 tests with both optional gates
off.

A real xcvu3p import additionally produced a byte-deterministic, round-tripped
`out/phase7/xcvu3p.v1.bfgraph` with 28,226,432 vertices and 130,278,682 directed
edges. At the maintainer's direction, the unusually long full-scale CSR/CSC and
tile-directory pass was stopped after those checks passed, before the 13-input
scan and real query artifacts. The implementation and exact continuation
command are available, but that interrupted portion is not claimed as executed
evidence. No HIP or GPU test was attempted.

## Phase 8 passed under the CPU-only acceptance policy

Phase 8 adds shared GPU foundations rather than an SSSP engine:

- deterministic maximal per-row CSR destination-tile runs and per-column CSC
  source-tile runs with 64-bit host metadata and deep coverage, boundary,
  maximality, and endpoint-ownership validation;
- a checked 32-bit relaxation-hot staging layout containing owner tiles, both
  sparse orientations, duplicated float weights, and both run orientations,
  while intentionally omitting stable edge IDs, provenance, and vertex
  metadata;
- exact component-by-component resident graph memory reporting before
  allocation, plus deterministic bitwise deep-layout comparison;
- retained CSR/CSC run-lane-mask materialization and an exhaustive small-graph
  proof that run-level admission matches endpoint-by-endpoint admission;
- one fixed-width API vocabulary for the three future engines and three control
  modes, with validated trivially copyable run options, controller, status,
  counters, stop reasons, and self-identifying result records;
- exact controller/status terminal and error invariants, explicit copied-record
  ABI sizes and alignment, plus distinct host-synchronization and
  controller-copy work counters;
- checked workspace byte estimation plus geometric retained capacities,
  generation-tagged stale-lease rejection, mandatory active-prefix clearing,
  one shared run-mask allocation, and one shared large scratch capacity for the
  active engine rather than one large allocation per engine, with separate
  device, pinned-host staging, and combined byte totals; and
- a HIP-gated runtime with checked error handling, stream/event/buffer RAII,
  asynchronous transfers and clears, device-event and wall timers, immutable
  fixed-width CSR/CSC views, a one-upload resident graph owner, pinned retained
  query staging, a no-allocation query-preparation path, and optional retained
  instrumentation storage. Resident plans deeply validate raw tile runs before
  allocation, observing upload readiness releases the full graph staging, and
  leases enforce a single preparation/result/retirement stream.

Passing CPU evidence is 8/8 CTests in `build/phase8-cpu` from
`ctest --test-dir build/phase8-cpu --output-on-failure` (13.41 seconds). This
includes the focused `bfnew.device_layout` and `bfnew.workspace` CTests plus
the unchanged earlier CPU tests. The fixtures cover empty CSR rows and CSC
columns, internal/adjacent/long/spill tile runs, parallel edges, spatial
reordering, multi-source/multi-target workspace planning, checked 32-bit
overflow rejection, repeated-capacity reuse, engine switches, and stale
leases. No full-graph search, HIP execution, SSSP kernel, or performance
measurement is claimed.

The HIP-only `bfnew.device_transfer` source checks bit-preserving round trips
for its tiny hand-checked layout, the existing Phase 5 spatial/spill fixture,
and an empty spatial graph. It also checks a two-source/two-target query payload,
controller/tile/run masks, stable allocations across repeated preparation,
cleared status and instrumentation, stale-lease rejection, and scratch reuse
across engine kinds. A tiny test-only validation kernel checks CSR/CSC run
bounds, ownership, maximality, and run admission on the raw device views; it
performs no SSSP relaxation. That HIP code was not compiled or executed
locally. The source additionally rejects corrupted raw run plans, exercises a
download after upload staging is released, checks split/combined workspace byte
reports, and rejects a result copy on the wrong stream. Only its C++20 syntax
was checked against a temporary no-runtime stub. Its AUP run and allocation/
copy trace, as well as a bounded-real device round trip, remain deferred by the
maintainer's policy and are not passing evidence or performance evidence.

All HIP copies are stream ordered. Internal query staging is page locked, but
resident-graph transfers and caller-provided status/instrumentation destinations
may be pageable. Those operations are therefore not claimed to overlap host
execution unless their host endpoints are explicitly page locked.

The runtime cannot identify the stream used by a kernel launched directly from
a raw `DeviceWorkspaceView`; callers must keep those consumers on the lease's
bound stream until an execution-registration API is implemented.

## Phase 9 passed under the CPU-only acceptance policy

Phase 9 implements only standalone synchronous Jacobi CSC pull:

- a checked scratch layout with exactly two graph-sized float distance
  columns, plus a typed view that exposes distinct immutable-read and
  mutable-write pointers;
- a deterministic two-word fingerprint over the exact checked 32-bit hot
  graph/run image, compared against the resident upload when constructing a
  Jacobi engine so a same-shaped wrong graph is rejected; this is an accidental
  identity guard, not a cryptographic integrity mechanism;
- complete destination-owned Jacobi updates using incoming CSC, the unchanged
  preceding label candidate, ordinary float addition, strict `<`, and no
  atomic distance update;
- source seeding into both slots, no target-based early stopping, and a final
  no-change scan that leaves the two complete columns bitwise identical;
- CSC source-tile run admission materialized once per run from destination-
  owner and source-tile masks, with zero runs skipped in full and no per-edge
  tile admission lookup;
- one shared Jacobi controller transition for portable and HIP paths, including
  actual-round accounting, uniform slot swapping, per-lane mask semantics,
  explicit convergence/maximum/error stops, byte-preserving terminal no-ops,
  and final slot identity taken from the controller;
- portable models of independently selectable persistent cooperative,
  per-round host-poll, and chunked host-poll control at `K = 2,4,8,16,32`,
  including exact submitted-dispatch, host-check, semantic controller-
  observation, and synchronization accounting; and
- a HIP-gated `JacobiPullEngine` source with ordinary initialization plus
  ordered round/advance kernels and one persistent cooperative query kernel.
  The persistent source performs GPU initialization, grid-stride destination
  work, block-reduced changed state, designated-owner controller transitions,
  uniform grid barriers, and final status without host convergence polling.

The HIP source queries occupancy on the real ordinary and persistent Jacobi
kernels with their dynamic shared-memory requirement. It rejects fixed
blocks-per-WGP requests above that result and does not reuse the Phase 6 probe
ceiling. With instrumentation disabled, its only relaxation-global atomic is
the at-most-once-per-block controller changed-flag update; optional counters
add block-reduced statistic atomics. Distances are destination-owned and have
no atomic-minimum or CAS update. It reports GPU-event and host-wall fields,
cooperative grid/occupancy fields, engine/control identity, rounds, CSC edge
and useful-decrease counts, active destinations, mask work, dispatches, host
checks, controller copies, and synchronizations. Queue and distance-atomic
counters remain zero. Mask-operation counting is Debug-only; Light retains the
round/edge/decrease/active-destination counters without that extra per-run
increment. Unlike the portable observation count, the HIP controller-copy
counter is physical: initial H2D plus every poll D2H plus terminal D2H. The
reported GPU-event value is a device-timeline span. In host-poll modes it
includes stream-idle gaps between observation and re-enqueue, so it is not
summed kernel-active time and cannot support a pure-GPU cross-control
comparison without deferred profiler kernel durations.

Portable passing evidence is 9/9 CTests in `build/phase9-cpu` from
`ctest --test-dir build/phase9-cpu --output-on-failure` (0.62 seconds),
including `bfnew.jacobi_pull`. The focused test runs 13
small deterministic fixtures through seven control configurations each and
compares every result with Dijkstra on the same selected-tile induced graph.
Representable cases require bitwise agreement; three fixed random graphs use
the established four-ULP reporting comparison; the focused run observed a
maximum difference of 0 ULP for those fixed seeds. The fixtures cover the
Phase 5 core/spatial/equal-tie/zero-cycle behavior, a long chain, disconnected
vertices, parallel edges, canonical multiple sources, an admitted path that is
more expensive than an excluded global path, a bounding-box miss, and three
random seeds. Additional assertions cover exact run admission, source zeros,
two-column convergence identity, final parity, K=32 queued no-ops,
maximum-round exhaustion in all three controls (including a full `K=32`
final chunk), error/miss separation, scratch overflow and
nonaliasing, deterministic/content-sensitive graph fingerprints,
validator-clean terminal errors from malformed controller fields,
None/Light/Debug counter behavior, and a structural HIP source audit.

The HIP-only `bfnew.jacobi` CTest source contains its own 13-case matrix for all
three controls and all five K values, plus occupancy-derived and fixed one- and
two-blocks-per-WGP cooperative grids on the core fixture. It also checks
bounded Dijkstra agreement, result parity, wrong-engine rejection,
maximum-round behavior, zero distance-atomic counters, and nonnegative timing
fields. It also rejects a same-shaped resident graph whose one changed weight
produces a different identity fingerprint. That source and test passed only a
host-side fake-declaration syntax check; no HIP compiler or GPU executed them.
This is not passing GPU evidence, and no real occupancy, trace,
PMC, timing, bounded-real, or full-graph claim is made. Exact eventual commands
are in
`docs/PHASE9_GPU_VALIDATION.md`; all GPU work remains deferred until the
combined campaign after the implementation phases. The current whole-matrix
test also cannot isolate ordinary dispatch latency or Jacobi grid-barrier cost;
the Phase 6 probe-kernel barrier result would not be a substitute.

## Implemented Phase 10 standalone dense chaotic push

Phase 10 adds a portable dense-push semantic/controller implementation and an
optional HIP-gated engine source. The engine keeps exactly one unsigned
IEEE-754 distance word per vertex and allocates no second label column,
predecessor, frontier, queue, or worklist. Its accepted domain is canonical
positive zero, finite nonnegative floats, and positive infinity; unsigned
integer minimum is therefore equivalent to float minimum. Negative zero,
negative values, NaNs, and encodings above positive infinity are rejected and
covered by CPU tests.

Every executing round is a complete admitted outgoing-CSR scan. Admission is
materialized once per source/destination tile run and zero-mask runs are
skipped in full. The portable model reloads the current source label for every
edge and applies the same strict unsigned minimum as the device CAS loop.
Forward, reverse, and alternating schedules may require different round counts
but must converge to the same bounded-Dijkstra bits, making the intended
chaotic/asynchronous semantics explicit without claiming an ordered serial
schedule.

The shared dense controller transition counts complete scans, never swaps its
single distance slot, gives no-change convergence precedence at the exact
round limit, and converts malformed state to a validator-clean terminal error.
Persistent, per-round, and full-`K` chunked protocols are independently
modeled. A `K=32` chunk retains queued no-op round/advance pairs after device
`done` while leaving actual completed rounds unchanged.

The HIP-gated `DenseChaoticPushEngine` source implements:

- CSR source-row grid-stride scans and endpoint-intersection run masks;
- CAS-compatible atomic source loads plus CAS-loop destination minima, with no
  ordinary distance read racing an atomic update;
- one block-reduced changed publication per changed block;
- ordinary initialization, ordered round/advance pairs, and final status;
- one persistent cooperative query kernel containing initialization, run-mask
  setup, full scans, controller transitions, and status;
- actual-kernel occupancy checks for ordinary and cooperative grids; and
- None/Light/Debug instrumentation for edges, decreases, active vertices,
  full scans, atomic attempts/successes, mask work, changed publications, and
  unique admitted high-contention destinations.

The HIP source and its complete tiny-fixture GPU CTest have not been compiled
with HIP or executed on a GPU. A host-side fake-declaration syntax audit and a
CPU structural source audit are source-level checks only. They establish no
device correctness, occupancy, profiler, timing, bounded-real, or performance
evidence. Deferred commands are in `docs/PHASE10_GPU_VALIDATION.md`.

Bounded CPU evidence is 10/10 CTests in `build/phase10-cpu` (0.80 seconds).
The focused sanitizer build/test also passed with AddressSanitizer and
UndefinedBehaviorSanitizer in `build/phase10-asan`. The focused test covers
the 13 Phase 9 fixtures plus high-fan-in and hub-fan-out cases through seven
control configurations, bounded Dijkstra comparison, adversarial schedules,
repeat stability, maximum-round stops in all controls, exact `K=32` no-op
accounting, atomic-bit restrictions, run admission, and counter-level
instrumentation. The fixed seeded-random maximum observed difference is
0 ULP.

## Implemented Phase 11 standalone active-frontier push

Phase 11 adds a portable frontier-push semantic/controller implementation and
an optional HIP-gated engine source. It is a separate engine from dense push.
Its retained scratch has one 32-bit atomic-domain distance word per vertex, two
bounded 32-bit vertex queues, and one 64-bit enqueue generation per vertex.
The default queue capacity is exactly the vertex count; a smaller explicit
capacity is retained only to exercise the required overflow path. The checked
layout aligns the generation array, rejects empty/oversized capacities and
undersized or misaligned storage, and never writes past a full queue.

The initial frontier is every canonical source exactly once. All source and
target terminals stay in one multi-source/multi-target query rather than
becoming separate searches. Each current frontier entry owns one outgoing CSR
row, and all enqueue-generation words are reset to zero at query initialization
before round one uses generation value one. The engine reuses the Phase 8
destination-tile run masks. A successful strict unsigned-float atomic minimum
activates the destination for the next round. A per-destination atomic
generation exchange allows exactly one queue claim in that round; later
successful improvements update the distance while counting as duplicate
suppressions. This preserves the worklist invariant without sacrificing a
later, better label. Queue exhaustion produces an explicit validator-clean
`queue_overflow` stop, never silent truncation.

The shared frontier transition increments only actually executed rounds,
swaps read/write queues, clears the recycled size, and distinguishes
empty-frontier convergence, exact maximum-round exhaustion, overflow, invalid
controller state, and device failure. The single distance array never swaps;
both distance slots and the final status slot remain zero. A bounding-box miss
is published only after convergence proves that a requested target is still
unreachable.

The portable model covers persistent, per-round, and complete-`K` chunked
control for `K = 2,4,8,16,32`. A queued pair after `done` is a no-op, so a full
final chunk does not corrupt round counts or queue parity. The HIP-gated source
implements one initializer, ordinary round/advance pairs, one finalizer, and a
single cooperative persistent query kernel. It uses CAS-compatible source
loads and destination minima, atomic generation exchange, atomic queue
reservation, block-reduced statistics, uniform grid barriers, and occupancy
queries on the actual frontier kernels. It deliberately does not implement
prefix-sum edge balancing, virtual warps, a special high-degree path, wave
aggregation, or adaptive scheduling.

The simple mapping deliberately leaves two eventual combined-campaign
measurement questions open: unequal outgoing row lengths can imbalance
frontier threads, while high-fan-in destinations and the global next-size
reservation can concentrate atomic contention. Phase 12 can supply the
shootout interface, but no unexecuted device path establishes a bottleneck or
speedup and no GPU run is implied during that phase.

Light instrumentation records edges, decreases, active frontier vertices,
executed/empty/small frontier rounds, and maximum queue size. Debug adds run
mask operations, atomic attempts/successes, queue claims, duplicate
suppressions, and overflow events. None suppresses algorithm counters. The
portable output retains the exact current-frontier size for every round and
checks its sum against aggregate active vertices. HIP retains aggregate
`active_vertices` and `active_lane_rounds`; it does not export a per-round
device trace. The current fixed ABI is eight-byte aligned with
`DeviceWorkStatistics` at 168 bytes and `GpuSsspResult` at 224 bytes.

Bounded CPU evidence is 11/11 CTests in `build/phase11-cpu-release` (0.09
seconds). The focused binary reports a maximum seeded-random difference of
0 ULP. Its 17-fixture matrix covers sparse-chain, rapid-expansion, high-fan-in,
hub-fan-out, repeated-improvement, zero-cycle, multiple-source, disconnected,
bounded-miss, invalid-empty, capacity-overflow, and maximum-round cases; all
three controls and all five K values; bounded Dijkstra agreement;
repeat/control bit stability; exact queue and counter invariants; and a
structural audit of the HIP source.

The optional `bfnew.frontier_push_gpu` CTest source is implemented and wired to
the HIP build. It repeats the tiny control matrix and adds instrumentation,
multiple real-occupancy-bounded cooperative grids, maximum-round, overflow,
and empty-initial-frontier checks. The final engine source passed a host C++20
fake-HIP declaration syntax check with only expected fake-stub warnings, and
the HIP test passed a warning-clean host C++20 syntax check. Neither has been
compiled with HIP or executed on a GPU. Source inspection is not device
correctness, occupancy, trace, profiler, timing, bounded-real, or performance
evidence. Every GPU run
remains deferred until the combined campaign after all implementation phases;
the eventual commands and evidence checklist are in
`docs/PHASE11_GPU_VALIDATION.md`.

## Implemented Phase 12 controlled single-query shootout infrastructure

Phase 12 adds a reproducible evidence model and comparison harness around the
three already separate engines. It is not a fourth algorithm, adaptive switch,
hybrid, or measured optimization. `ShootoutTuning` gives every result a stable
identity across Jacobi pull, dense chaotic push, and frontier push; persistent,
per-round, and chunked `K = 2,4,8,16,32` control; block sizes 128, 256, and 512;
and occupancy-derived or explicit persistent blocks-per-WGP policies. Runtime
kernel-limit records preserve illegal and unavailable configurations with an
explicit reason instead of silently shrinking the matrix.

The manifest binds graph and query fingerprints, requires at least 1,000
representative candidates, and stratifies selected-region vertices and edges,
fanout, source count, and deterministic Jacobi-pilot rounds into combined
quantile bins. Seeded selection and scheduling are stable, and retained timing
interleaves query/configuration order for every repetition. Correctness groups
all configurations for one query and assigns truthful ordinals so it builds
the bounded-Dijkstra oracle exactly once per query. The pilot round
count is selection metadata, not performance evidence. Named sparse-wavefront
and dense-frontier cases are separate built-in GPU-runner workload identities;
they do not satisfy or mix with the real-corpus minimum.

Evidence is deliberately split into warmup, correctness, unprofiled timing,
Debug algorithm counters, dependency trace, and PMC stages. Complete
bounded-Dijkstra correctness for every selected query/configuration pair gates
all later evidence. Correctness downloads final distances outside timing.
Only selected tile ranges are copied for the local oracle comparison.
Timing requires instrumentation None and status-only transfer, preventing the
different one- versus two-column engine layouts from creating incomparable
graph-sized result-copy costs. Preparation, SSSP device timeline, result
transfer, and end-to-end wall time are distinct fields. Trace and PMC evidence
cannot be relabeled as ordinary timing.

The HIP-gated executor dispatches exactly the three existing engines over one
resident graph/workspace/stream. Every non-correctness stage uses the engines'
status-only path; timing rejects retained-workspace allocation growth and any
graph-sized label result. Pilot edge metadata uses a one-time CSR tile-pair
index instead of an all-vertices/all-edges scan per query. The two-stage real
pilot computes cheap features over the eligible corpus, chooses a domain-
separated stratified shortlist of at most four times the requested count, and
runs status-only Jacobi only on that shortlist before final five-dimensional
selection. Correctness constructs one selected-tile/admitted-run local Dijkstra
oracle per query rather than a graph-sized CPU label vector. The driver persists
the runtime-resolved catalog and requires later stages to reload it rather than
reconstructing tunings from defaults. A separate Debug legality pass preserves
rejected/unavailable counter cells instead of manufacturing zeros.

The schema distinguishes unavailable, not-applicable, and measured values.
Deterministic summaries cover nearest-rank P50/P95/P99 wall and device-timeline
distributions, throughput, work and atomic counters, actual grids/occupancy,
L2 and memory-unit behavior, instruction mix, and up to 50 P99-tail records per
configuration with their workload features. TSV manifests/samples and JSON
reports use stable order. Input-fingerprint, run-kind, configuration,
instrumentation, sample-identity, status, and completeness checks fail closed.

The long-tail replay plan is a separate global subset: it covers each available
tail-metric/engine/control stratum, fills remaining slots by severity, deduplicates
stable case IDs, and stops at 48. Every row repeats the full replay policy and
input identity. Batched profile processes execute one unrecorded status-only
warmup per case and bracket each measured replay with named begin/end marker
kernels. The profiler importer requires a one-to-one mapping from those stable
case IDs to normalized metric groups.

Without an explicit conclusions TSV, the host report intentionally leaves all
requested conclusions pending:

1. persistent versus chunked control;
2. best K per ordinary engine;
3. maximum cooperative occupancy helping or hurting;
4. dense pull versus sparse frontier regions;
5. dense chaotic push beyond diagnostic use;
6. the cause of expensive tail queries; and
7. eliminated per-round polling time.

Recommended defaults are also `pending measured GPU campaign`. Every engine,
control, K, grid, and block toggle remains configurable. This prevents the CPU
semantic models or unexecuted HIP sources from being presented as a performance
recommendation. After a future campaign, the optional TSV requires all seven
workload-matched question rows and configurable recommendation rows whose
configuration IDs and evidence references resolve exactly against the report's
supplied evidence inventory.

Bounded CPU evidence: the Release build in `build/phase12-cpu` passed 12/12
CTest cases in 1.10 seconds. The focused `bfnew.shootout` binary passed and
explicitly reported that it collected no GPU timing evidence; its ASan/UBSan
run passed 1/1 in 0.31 seconds. These tests cover deterministic
catalog/manifest/schedule/report behavior and adversarial tiny correctness
without performing a real-corpus or large-graph search.

Neither the real `logicnets_jscl` query artifact nor a representative manifest
was produced locally. No Phase 12 HIP source or driver has been compiled with a
HIP compiler, and no GPU correctness, occupancy, timing, trace, PMC,
long-tail, throughput, conclusion, or default evidence was collected. The
future combined-campaign procedure is documented in
`docs/PHASE12_GPU_VALIDATION.md`.

## Implemented Phase 13 deterministic overlap planning and workspace model

Phase 13 adds host preparation and accounting only; it does not add a batched
relaxation kernel. A graph-bound `SelectedRegionIndex` deep-validates graph/run
metadata once, coalesces exact ordered tile-pair edge counts, and answers
selected-region or union vertex/edge estimates without scanning every graph
vertex and edge for each query. Batch features preserve query identity,
expansion generation, independent source/target counts and owner tiles, actual
selected tiles, and exact selected-region estimates. Canonical QueryId ordering
makes the complete plan and TSV independent of RouteQuery input permutation.

The greedy planner emits the required 32-, 16-, and 8-lane comparison family.
It selects large anchors deterministically, then maximizes Jaccard and terminal
tile overlap while minimizing exact projected union inflation, edge/vertex
work, new tiles, and generation/source/target-count differences. Configurable
minimum-Jaccard and maximum-inflation fractions are compared exactly. Every
query appears once; singleton and other partial remainders occupy a contiguous
low-lane prefix and retain explicit padding. Deep validation rebuilds the
frozen greedy result and rejects changed identities, masks, estimates,
duplicates, omissions, or padding.

The reusable batch-device description retains independent flattened terminal
slices, QueryIds/generations, selected counts, sorted union tiles, exact
per-tile lane masks, and selected vertex ranges. It implements both Phase 13
run alternatives:

- retained CSR/CSC mask arrays with touched-run ledgers that clear only the
  preceding batch's nonzero entries; and
- sorted compact nonzero `(run_id,lane_mask)` descriptors with exact
  per-union-vertex CSR/CSC offsets.

Compact vertex storage has an actual reusable per-tile bias map, with
`compact_vertex = global_vertex - bias[owner_tile]` for selected tiles. Cold
zero-initialization and warmed touched-entry clear/write traffic are accounted
separately. A reusable run image is representation-bound after its first
preparation, preventing an unmodeled retained/descriptor switch.

Both compute endpoint admission once per maximal run. Bounded proof tests show
their CSR and CSC masks agree exactly with endpoint intersections and admit the
same per-lane edges, including spill and cross-tile runs. This removes repeated
per-edge tile admission work from the future kernel interface; it does not yet
measure a GPU benefit.

Checked workspace estimates compare full graph-sized and compact union-tile
vertex storage crossed with both run representations. Component bytes,
selected/wasted lane vertices, reset and preparation writes, active/zero runs,
allocation reuse, and budget-limited concurrency remain separate. A retained
reservation grows component capacities geometrically and reuses them across
batches. On the bounded synthetic acceptance fixture the provisional choice is
full graph-sized vertex-major lanes plus retained run masks: it avoids compact
label mapping and has the simpler reusable preparation path. Fixture budget,
bytes, preparation counters, and bounded host build times are retained in
`docs/PHASE13_WORKSPACE_DECISION.md`.

The Phase 7 record supplies only `V = 28,226,432`, not the missing real query
corpus. Checked analytical distance-label bytes at width 8/16/32 are
903,245,824 / 1,806,491,648 / 3,612,983,296 for one slot and
1,806,491,648 / 3,612,983,296 / 7,225,966,592 for two. An explicitly
illustrative 64-GiB-capacity, 8-GiB-resident, 4-GiB-reserve scenario is not a
runtime memory observation or allocation proof. Real union/run distributions,
free memory, build time, concurrent capacity, and all HIP/GPU performance
remain unmeasured, so the production choice is deferred.

Bounded CPU evidence is `14/14` Release tests passing in `0.05` seconds. A
separate focused Phase 13 invocation passed `2/2` in `0.26` seconds, and its
ASan+UBSan run passed `2/2` in `0.45` seconds. The focused tests perform no full graph scan,
real-corpus execution, HIP compilation, or GPU run.

## Implemented Phase 14 overlapping batched Jacobi semantics

Phase 14 adds a separate `batched_jacobi_pull` path for widths 1, 8, 16, and
32. It is a correctness and measurement foundation, not a demonstrated
optimization. The standard overlap planner stays at widths 32/16/8; explicit
width one provides the bitwise standalone-Jacobi baseline.

The portable model implements:

- two vertex-major distance slots with contiguous query lanes;
- independent offset source/target slices and exact per-lane multi-source
  seeding in both slots;
- selected-range destination admission and endpoint-exact incoming CSC
  source-tile run masks;
- one shared physical edge-record traversal followed by independent strict
  float relaxation for every active lane;
- both retained masks and compact nonzero run descriptors;
- enabled or disabled per-lane convergence through the shared Jacobi
  controller, defaulting to enabled;
- complete no-change writes before lane freeze, which preserves bitwise slot
  equality through later batch-wide swaps;
- persistent, chunked-host-poll, and per-round-host-poll semantic protocols;
- post-convergence per-lane all-target reached/miss classification; and
- checked shared-edge, lane-edge, run, utilization, padding, tail-work,
  union-inflation, modeled selected-device reset, and source-seed accounting.

The bounded validation matrix retains lanes that converge immediately, after
one or several changing rounds, and with an unreachable target. It compares
every valid lane against an independently induced bounded Dijkstra problem,
compares enabled and disabled convergence results, exercises independent and
multi-source lanes plus padding, and checks an early lane's two slots after
several later global swaps. Width one follows the standalone CSC order and is
checked bitwise against `run_host_jacobi_pull`.

The separately gated `hip::BatchedJacobiPullEngine` owns a batch-specific
reusable workspace rather than encoding a batch as one `RouteQuery`. The
workspace retains flattened terminals and their offsets, selected ranges and
packed offsets, tile masks, a device-only dense CSC run-mask image,
controller/status,
optional statistics, per-lane convergence rounds, and two full graph-sized
distance slots. Preparation, execution, result transfer, and retirement are
bound to one stream. Production host preparation builds no run image. The
initialization wave for each selected destination cooperatively materializes
its CSC masks from destination/source tile admission, writing zero and nonzero
entries before relaxation. This removes the per-batch graph-wide run-mask H2D
copy and clear without adding a kernel, atomic, or synchronization, while
preserving the hot round's single dense mask load. Retained host masks remain a
deep-validated compatibility path; the portable descriptor path remains
correctness evidence for Phase 13 equivalence.

One wave32 owns a destination and maps wave lanes to query lanes. Nonzero run
masks reuse the incoming edge source and weight across those lanes. The
default leaves those fields as compiler-visible uniform loads. A separately
selectable explicit `__shfl` broadcast path exists only for the required later
comparison; it is not selected and no benefit is claimed. Ordinary kernels
pair relaxation with controller advancement. The persistent variants retain
device-side convergence and uniform cooperative-grid barriers. Reached/miss
classification is device-side after convergence.

Across all three standalone and batched HIP engines, terminal result-transfer
completion now also retires the same-stream lease. Normal full/status paths no
longer fence a second time, and compact-path paths release an already-complete
lease without another fence. Therefore `N` controller polls require `N + 1`
host synchronizations. Standalone and batched push controller copies remain
`N + 2`; batched Jacobi is `N + 1` because its initialization kernel now writes
the complete controller and no overwritten initial controller is uploaded.
Jacobi preparation also omits the unused union-tile upload and redundant status
clear.

Post-amendment local verification passed all 20 HIP-off Release tests, all 20
strict `-Werror` ASan+UBSan tests, and all 21 strict FPGA-Interchange-enabled
tests. HIP/ROCm is unavailable on the local machine, so these results do not
replace the required `gfx1151` HIP compile, device correctness, trace, or timing
campaign.

Phase 14 local evidence is the Release `build/phase14-cpu` suite passing all
`15/15` bounded CTests in `1.38` seconds. Its focused Phase 13+14 matrix passed
`3/3` in `0.02` seconds; `bfnew.batched_jacobi_pull` passed `1/1` in `0.01`
seconds in the final full run. The ASan+UBSan `build/phase14-asan` suite passed
all `15/15` in `2.73` seconds, including the Phase 14 CTest `1/1` in `0.20`
seconds. Local testing
does not include HIP compilation, GPU execution, a real query corpus,
uniform-versus-broadcast timing, register pressure, actual occupancy, L2/read
traffic, latency, throughput, or batching improvement. Those remain explicitly
deferred in `docs/PHASE14_GPU_VALIDATION.md`; none is encoded as zero or
inferred from CPU work counters.

## Implemented Phase 15 overlapping batched dense semantics

Phase 15 adds a separate `batched_dense_chaotic_push` path for widths 1, 8,
16, and 32. It is a correctness, accounting, and later-measurement foundation;
no batching speedup or device optimization has been demonstrated. The standard
overlap planner remains 32/16/8, with explicit width one as the standalone
dense baseline.

The portable model implements:

- one vertex-major unsigned-float atomic word per vertex/query lane;
- independent offset terminal slices and exact per-lane multi-source seeding,
  with no cross-lane source merge or destination update;
- selected source rows and endpoint-exact outgoing CSR destination-tile run
  masks, in retained and compact-descriptor forms;
- one algorithmic edge-record request followed by an independent
  atomic-compatible source load and strict destination atomic-min attempt for
  every admitted lane;
- complete admitted-edge scans with enabled or disabled per-lane convergence,
  defaulting to enabled;
- persistent, per-round-host-poll, and chunked-host-poll protocols for
  `K = 2,4,8,16,32`;
- post-convergence per-lane all-target reached/miss classification with no
  target early stop or frontier state;
- exact considered/visited/skipped run, atomic attempt/update, full-scan,
  active/tail/avoided-work, union-inflation, one-slot reset, and source-seed
  accounting; and
- separate valid-lane, configured-width, wave32, unused-wave, padded-lane, and
  edge-wave lane-capacity denominators.

The field named `csr_edge_loads` is a logical shared edge-record request. It is
not a physical cache-load count, L2 transaction measurement, or proof that the
compiler emitted one load. The edge-wave and lane-round denominators are exact
integer utilization terms, not measured GPU occupancy. Likewise, selected
one-slot reset bytes model the device initializer's required semantic writes
and exclude convenient nonselected initialization performed by a fresh
portable host vector. The shared `high_contention_destinations` field is a
standalone dense-engine diagnostic and is unavailable for batched Phase 15;
its zero value is not evidence that device contention is absent.

The bounded validation matrix shares the Phase 14 tiny graph but uses reverse
CSR order to retain lanes with first no-change scans at 1, 2, 6, 1, and 4.
It therefore includes two no-first-decrease lanes, a bounded unreachable lane,
short and several-scan paths, multi-source behavior, and explicit cross-lane
isolation. It checks both convergence settings in every control, retained/
descriptor parity, instrumentation separation, exact counter identities,
maximum-round behavior, independent bounded Dijkstra results, and width-one
bitwise agreement with `run_host_dense_chaotic_push`.

The separately gated `hip::BatchedDenseChaoticPushEngine` uses
`ReusableBatchedDenseWorkspace` and consumes a validated
`BatchDeviceDescription`. The retained workspace owns flattened terminals,
selected ranges, dense tile masks, exact retained CSR masks, controller/status,
optional statistics, convergence rounds, and one full graph-sized atomic word
array. This is the provisional Phase 13 full-vertex/retained-CSR choice; compact
device storage is not implemented or selected merely because portable
descriptor parity passes.

The HIP source maps one wave32 to a selected source row and query identities
to wave lanes; the wave services that row's outgoing CSR edge requests.
`BatchedDenseLoadStrategy::compiler_uniform` is the unmeasured default. The
selectable `explicit_wave_broadcast` variant exists only for a future
controlled comparison; it is not a retained optimization claim.
Ordinary scan/advance pairs and a persistent cooperative path are present in
source, with device-side final target classification. The HIP source has not
been compiled or executed locally.

Final Release evidence: `build/phase15-cpu` passed `16/16` bounded CTests in
`0.17` seconds. The focused `bfnew.batched_dense_chaotic_push` CTest passed
`1/1` in `0.01` seconds. Final ASan+UBSan evidence:
`build/phase15-asan` passed `16/16` in `1.82` seconds, including the Phase 15
CTest `1/1` in `0.14` seconds.

Local evidence includes no real query artifact, HIP compiler, GPU execution,
device correctness, uniform/broadcast measurement, registers, actual
occupancy, L2 reads/writes, atomic or write stalls, latency, throughput, or
batching benefit. All remain unavailable and deferred to
`docs/PHASE15_GPU_VALIDATION.md` for the combined campaign after all phases.

## Implemented Phase 16 overlapping batched frontier semantics

Phase 16 adds a separate `batched_frontier_push` path for widths 1, 8, 16,
and 32. It is a correctness, queue-invariant, accounting, and future-
measurement foundation; no GPU batching speedup or production configuration
has been demonstrated. The standard overlap planner remains 32/16/8, with
explicit width one as the standalone frontier baseline.

The portable model implements:

- one vertex-major unsigned-float distance word per vertex/query lane, with
  independent terminal slices and multi-source seeding;
- two bounded vertex queues and two per-vertex lane-mask arrays, where one
  entry can carry several active query lanes;
- shared-source initialization that merges only the queue entry and never the
  lane-private distance state;
- endpoint-exact outgoing CSR run admission in retained and compact-
  descriptor forms, with one mask intersection per run and reuse across every
  edge in that run;
- atomic-compatible source loads, strict lane-private destination minima, an
  atomic next-mask OR for successful lane bits, and one queue claim only on a
  zero-to-nonzero mask transition;
- explicit overflow with no out-of-bounds queue write, plus exact queue/mask
  round-boundary validation;
- enabled or disabled per-lane convergence without synthesizing frontier work
  for a lane whose mask is absent;
- persistent, per-round-host-poll, and chunked-host-poll protocols for
  `K = 2,4,8,16,32`;
- post-convergence per-lane target reach/miss classification; and
- separate shared-vertex/shared-edge, lane work, mask/queue atomic,
  deduplication, utilization, tail, reset, and overflow accounting.

The portable and downloaded correctness results expose every configured
vertex/lane word. Every padded-lane word and every nonselected word in a valid
lane is normalized to positive infinity, so retained stale capacity never
escapes through the public full-output path. The device initializer still
writes selected semantic cells only; host-side output normalization is not
reported as device reset traffic. The production preparation boundary also
rejects noncanonical terminal slices, duplicate valid-lane query IDs, and any
semantic terminal or metadata payload in padded lanes.

The zero-to-nonzero activity-mask transition replaces Phase 11's enqueue-
generation array for batch deduplication. A physical queue contains vertices,
not vertex/lane pairs. The exact number of unique destination/lane activations
minus physical queue claims is the number of queue entries saved by lane
merging. A successful update after a destination is already represented still
updates its lane-private label and mask but does not duplicate the vertex.

The enabled and disabled convergence modes intentionally have the same actual
frontier work. Even with controller-level per-lane completion disabled, a lane
with no next-mask bit receives no invented queue entry. Its first empty-next-
frontier round and later batch-tail rounds are recorded, while semantic work
avoided solely by toggling the option may correctly remain zero. Final selected
distance words must be bitwise identical between the modes. Grid-stride
initialization clears every configured lane's convergence record. Only clean
accepted continue, convergence, or maximum-round transitions may publish a
record; error, invalid-controller, and terminal no-op transitions do not.

The HIP-gated `BatchedFrontierPushEngine` and
`ReusableBatchedFrontierWorkspace` are distinct from the standalone,
Jacobi-batch, and dense-batch owners. The retained workspace adds two vertex
queues and two graph-sized activity-mask arrays to
the one-slot vertex-major distance state, along with per-lane terminals,
selected ranges, exact retained CSR masks, controller/status/statistics, and
convergence rounds. A zero queue-capacity request means one entry per graph
vertex; smaller capacity is only an overflow-validation seam. The Phase 13
generic one-slot estimate excluded queues and activity masks, so a checked
Phase 16 scratch calculation accounts for them separately.

The device source retains the Phase 11 one-thread-per-worklist-entry scheduling
policy. It does not add edge balancing, virtual warps, a special high-degree
path, wave-aggregated queue insertion, or adaptive frontier scheduling. It has
no fallback to dense chaotic push and no adaptive push/pull combination. The
first device workspace uses the provisional full-vertex/retained-CSR choice;
portable descriptor parity is not a compact HIP implementation or a production
memory decision. Ordinary execution reports its occupancy- and queue-work-
capped launch count through `ordinary_grid_blocks`. The current HIP result and
deferred HIP CTest do not expose a complete per-round queue-to-mask boundary
ledger; exact device ledger proof remains a future or manual Debug diagnostic
for the combined campaign. The existing hot-path controller, mask, capacity,
and overflow guards remain active.

Instrumentation levels are intentionally non-overlapping. `None` suppresses
device algorithm counters. `Light` reports aggregate work, sharing, and queue
high-water only. `Debug` additionally reports mask atomics, unique
vertex/lane activations, physical queue claims, entries saved by lane merging,
same-lane and total duplicate suppressions, and overflow events.

The bounded validation matrix covers the full width/control/convergence/K
matrix; exact retained/descriptor parity across the whole configured `V * W`
output image and terminal result; independent, multi-source, shared-source,
and padded lanes; positive-infinity normalization outside each lane's selected
region; mixed frontier durations and bounded misses; shared-destination merging,
repeated improvements, high fan-in, rapid expansion, and zero-weight cycles;
exact portable queue/mask/counter invariants; initialization and round
overflow; instrumentation separation; clean maximum-round versus error convergence
evidence; width-one standalone parity; cross-lane isolation; and independent
bounded Dijkstra comparison. Every padded and nonselected output cell is
positive infinity in both retained and descriptor runs.

Final bounded evidence is HIP-off. Release passed `17/17` in `1.56` seconds,
including Phase 16 `1/1` in `0.02` seconds. ASan+UBSan passed `17/17` in `3.03`
seconds, including Phase 16 `1/1` in `0.16` seconds. The deferred HIP test
passed strict-warnings host syntax and fake-HIP `__HIPCC__` syntax, including
its device probe. Phase 16 public headers passed strict host syntax, and both
production HIP translation units passed strict fake-HIP syntax. These checks
do not constitute HIP compilation or device execution. No browser, cloud
service, or large/full-graph test was used.

Local evidence includes no real query artifact, HIP compiler, GPU execution,
device correctness, actual occupancy, physical L2 or memory traffic, latency,
throughput, batching benefit, or comparison with the earlier batched engines.
All remain unavailable and deferred to `docs/PHASE16_GPU_VALIDATION.md` for
the combined campaign after all implementation phases.

## Implemented Phase 17 failed-lane expansion and all-query execution

Phase 17 adds a deterministic controller above all three batched engines. It
does not change their graph traversal kernels. The implemented path now:

- accepts all route queries, canonicalizes state by QueryId, plans active
  overlap batches, and consumes one compact per-batch reached/miss status;
- accepts reached lanes immediately, collects every missed QueryId from the
  completed pass, expands all failures, deterministically replans them, and
  restarts each from its complete original source set;
- evaluates explicit one-ring, fixed-larger-ring, doubled-margin, and hybrid
  small-first/doubling schedules without compiling a guessed default;
- preserves all terminal/source identity and increments expansion generation
  while discarding old distance/frontier/controller state;
- supports a measured maximum expansion count followed by one full-region
  fallback, or explicit `expansion_limit`/`region_stalled` failure;
- terminalizes only per-query generation, retry/expansion-count, or margin
  overflow as `identity_or_count_overflow` without aborting unrelated queries;
  aggregate telemetry overflow instead aborts the campaign fail-closed, and
  neither boundary can wrap;
- distinguishes unreachable full-region, identity/count overflow, engine
  failure, and successful completion, so a missed target is never success;
- retains deterministic retry traces, expansion histograms, failed-lane and
  retry utilization, repeated region work, actual retry/failed-batch edge work,
  final region size, host replanning/execution time, integer throughput, and a
  schedule-comparison fingerprint that includes a required caller-owned
  execution-configuration identity; and
- selects a schedule only from a complete comparable four-record evidence set
  with a nonzero workload/configuration fingerprint and query count, a valid
  campaign, and a unique best score; an evidence-score tie selects nothing.

The portable adapter runs the existing batched Jacobi, dense, and frontier
engines on every retry. It accepts retained masks or compact descriptors and
supplies graph-sized terminal lane projections for induced-region Dijkstra
checks. The bounded matrix covers all schedule/engine pairs plus multi/single
misses, dissimilar regions, long cross-tile and spill paths, multi-source and
multi-retry restart, fallback, explicit failure, global unreachable, input-
order determinism, maximum QueryId and generation, malformed statuses/work
evidence, execution-configuration binding, invalid-campaign and exact-score-
tie rejection, exact retry metrics, and retained/descriptor parity.

The HIP-gated `BatchedExpansionExecutor` binds each of the three resident
engine/workspace pairs. Its compact mode requires None instrumentation and
downloads one 48-byte `DeviceRunStatus` after finalization, with no graph-sized
labels or counter/trace/controller records. Its separate evidence mode retains
the existing richer status-only engine paths and preserves measured versus
unavailable work. Retry preparation rebuilds retained masks and invokes each
engine's ordinary selected-only reset/source seed. No new expansion kernel or
device allocation was added.

The HIP+FPGAIF-gated `bfnew_gpu_batched_expansion` driver provides the deferred
real-corpus boundary. It requires explicit graph/query/output, engine, and
schedule arguments, supports every documented schedule/control/grid/terminal/
transfer choice, uploads one resident graph, executes the entire artifact,
validates the canonical all-query ledger, and writes a fingerprinted
`bfnew.batched-expansion.v1` report with aggregate metrics, histogram,
per-query outcomes, selected tiles, and retry traces. Output publication is
fail closed and non-overwriting. Its execution-configuration identity binds
transfer/load choices into cross-schedule evidence, and aggregate telemetry
overflow aborts rather than becoming a per-query disposition. The driver has
no label-download correctness mode or statistical/profiler aggregation. Its
controller timer begins after resident-upload synchronization but with a cold
reusable workspace, so it includes first workspace growth and is not warm
final-performance evidence. The driver has not been HIP-compiled or run
locally.

Local evidence remains bounded and HIP-off. Strict C++20 source checks and the
focused Release and sanitizer gates pass; final full-suite timing is recorded
in `docs/CPU_RESULTS.md`. Strict host and fake-HIP syntax cover the deferred HIP
test and changed device sources, but those checks are not a HIP compiler or GPU
run. Real `logicnets_jscl` all-query rates, expansion distributions, schedule
comparison, device correctness, occupancy, traffic, latency, and throughput
remain unavailable. The exact future procedure is
`docs/PHASE17_GPU_VALIDATION.md`, and no schedule is recommended from the tiny
synthetic fixture.

## Implemented Phase 18 compact reconstruction and result transfer

Phase 18 adds a post-relaxation result layer; it does not add predecessor state
to any hot engine. The implemented portable path:

- emits one fixed 28-byte summary per canonical target with distance,
  reachability, selected-source validity/identity, edge-count path length, and
  explicit reconstruction status;
- retains only path-sized global vertex IDs, actual final path labels, and
  stable logical edge IDs for complete targets;
- traverses incoming CSC tight edges in stable-`EdgeId` order using exact
  ordinary-float addition, path-local cycle detection, and real backtracking;
- handles zero-weight cycles, lower-ID tight dead ends, canonical multi-source
  termination, duplicate terminal maps, and zero-edge source targets;
- filters every candidate by final selected-tile membership instead of relying
  on stale/unselected label contents;
- binds each low-level payload to QueryId and expansion generation before
  workspace reuse, emits one for every clean valid lane, discards retry-miss
  generations only after proving at least one target unreachable and all
  remaining targets complete or unreachable, retains classified terminal
  full-region misses, and rejects all-complete or reconstruction-failure
  misses as engine failures along with stale, padded, duplicate, or wrong-
  generation payloads;
- validates source/target termination, simple vertices, edge continuity,
  compact-label tightness, exact reported cost, and selected-region membership
  without a graph-sized label download; and
- reports checked summary, vertex, path-label, edge-ID, and total compact
  serialization bytes, charging checked-u32 device vertex and edge-ID words at
  four bytes each even though host edge IDs widen after transfer; physical D2H
  remains a separate HIP evidence/trace claim whose compact payload/status/
  error subtotal excludes controller polling and whose checked overall total
  adds separately reported controller-poll bytes.

The portable Phase 17 adapter can capture compact payloads during the terminal
batch. `extract_host_compact_paths` consumes those payloads or reconstructs
from the bounded diagnostic terminal projection, generates explicit terminal-
failure summaries when labels are unavailable, and releases every graph-sized
correctness image. Its returned results contain no distance matrix.

The optional HIP path reconstructs from the exact final distance slot after
the existing batch finalizer and before workspace retirement. It transfers
target summaries followed by exact compact path arenas and does not download a
`V * W` label image in production mode. Relaxation remains distance-only. The
source has not been compiled with a HIP compiler or executed on a GPU locally,
so this is implemented-but-unvalidated device behavior, not a transfer or
performance result.

The implemented no-congestion reference layer also:

- assembles explicit planning, SSSP, geometric region growth, controller/
  orchestration, reconstruction, and transfer stages after capacity is grown;
  requires their exact `warm_all_query` sum when every named host interval is
  measured, while permitting a separately measured enclosing warm interval
  when asynchronous HIP leaves named host stages unavailable rather than
  measured-zero; and keeps a separate first capacity-growing `cold_execution`
  whose sum with artifact load and graph upload is `cold_pipeline`, without
  equating cold and warm execution;
- validates host/device timing evidence, rejects numeric values labeled
  unavailable, and uses checked exact host sums without inventing aggregate
  device time;
- partitions all target statuses; models canonical final serialization with
  48-byte status records, 28-byte summaries, checked-u32 vertices/edge IDs, and
  float labels without claiming physical D2H; and separately preserves checked
  cumulative HIP compact payload/status/error subtotal, controller-poll count/
  bytes, and overall D2H evidence when supplied, requiring zero polls for
  persistent control; and
- selects an input-order-independent SplitMix64 QueryId sample from a recorded
  seed, compares it with unbounded multi-source Dijkstra, and reports
  absolute plus nearest-rank P50/P95/P99/maximum cost/path-length inflation,
  including complete targets from mixed terminal results and the explicit
  positive-infinity convention for positive bounded cost over a zero baseline.

Quality inflation remains observation only. It is not an expansion trigger and
does not replace exact Dijkstra correctness on the final selected subgraph.

The bounded CPU fixture covers stable equal and parallel paths, zero-cycle
backtracking, fractional/zero weights, target-is-source and multi-source cases,
duplicate terminal identity, an excluded cheaper global route, long and spill
paths,
retry generations, full-region unreachable and engine failure, compact byte
accounting, timing-ledger malformed/unavailable/overflow cases, deterministic
quality sampling, malformed reached-status/incomplete-payload normalization,
exact diagnostic comparison, determinism, padded lanes, and widths 1/8/16/32
across all three portable batched engines. Final evidence is HIP-off Release
`19/19` in `2.02` seconds, including `bfnew.compact_paths` `1/1` in `0.10`
seconds; ASan+UBSan `19/19` in `2.82` seconds, including
`bfnew.compact_paths` `1/1` in `0.16` seconds. Strict host source/public-header
syntax and fake-`__HIPCC__` production, CLI, and deferred-test syntax passed
warning-clean.

The end-to-end no-congestion campaign and its required stage ledger are
specified in `docs/PHASE18_GPU_VALIDATION.md`. The absent representative query
artifact, HIP compiler, GPU execution, transfer trace, stage measurements,
all-query throughput, and representative-corpus path-quality results remain
unavailable.

The frozen path-campaign interface accepts lane widths 1/8/16/32; width 1 is
the scalar baseline. It fails closed unless the cold execution, each additional
warmup, and every measured repetition provide measured SSSP, reconstruction,
and result-transfer GPU-event evidence. Missing evidence is never reported as
measured zero or silently removed from the sample.

## Implemented Phase 19 final audit

Phase 19 adds a canonical evidence audit rather than an optimization. It
exhaustively inventories the portable and HIP-facing standalone engines,
control modes, shootout/profiler surface, overlap planner and workspace model,
three independent batched engines and widths, expansion schedules, compact
reconstruction, and the no-congestion end-to-end boundary. Every feature is
assigned exactly one of four evidence classifications. The 56-row local
inventory partitions them 36/0/12/8, with no class-2 row.

The final local snapshot deliberately has no
`implemented-and-performance-profiled` SSSP feature. Portable Release and
sanitizer correctness are class 1. Implemented but locally uncompiled/unrun HIP
engine, batch, expansion, reconstruction, and campaign sources are class 3.
Representative performance/profile results and recommendation questions are
class 4. These scopes remain separate: host correctness, structural source
inspection, fake-HIP syntax, bounded timing, work counters, and analytical
memory sizes do not become GPU-performance evidence.

The audit carries 27 canonical comparison rows spanning every required
engine-specific control, ordinary `K`, and persistent-grid comparison plus
batch width, planner threshold, expansion schedule, latency, throughput,
cold/warm end-to-end,
miss/expansion/fallback, quality, memory, synchronization/copy, and profiler
comparison. It also carries the seven explicit Phase 19 questions and ten
typed trace/PMC metric rows. An unavailable row contains no numeric value,
configuration selection, or evidence reference.

The `.phys` and `.netlist` generation inputs exist locally, but the required
versioned `logicnets_jscl.padding1.v1.bfqueries` artifact does not. HIP
compile/device execution, representative-corpus evidence, accepted trace/PMC
evidence, a matched CPU baseline, and normalized Phase 18 provenance/tail
attribution are also absent. The correct final result is therefore
`insufficient_evidence`: no performance comparison or project question is
measured, no production configuration is recommended, all toggles remain
configurable, and the hybrid-experiment decision is deferred.

The validator fails closed on missing, duplicate, reordered, or unknown IDs;
invalid classifications/evidence states; unavailable rows carrying payload;
inconsistent evidence identities; premature performance-profiled/measured
states; malformed configuration shape; duplicate support references; an
unresolved unique-winner attestation; and unstable serialization. Exact local
validation evidence is recorded in
`docs/CPU_RESULTS.md`, and the full comparison is
`docs/PHASE19_FINAL_AUDIT.md`.

## Excluded future algorithms

Adaptive push/pull switching, disjoint-region batching, an exit-edge global
certificate, and congestion updates are not planned in the replacement
campaign.

Phase 19 does not authorize congestion, an adaptive hybrid, or an
evidence-free default. Optional HIP/GPU, profiler, timing, bounded-real, and
full-graph evidence remains a separate maintainer campaign, not an unfinished
implementation phase.

An item may move out of this list only after its implementation phase has
passing source-level and test evidence.
