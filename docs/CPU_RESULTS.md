# CPU Results

## Current status

Phase 5 adds CPU correctness algorithms but no performance experiment or
algorithmic optimization. On deterministic synthetic fixtures, the Dijkstra
oracle and synchronous weighted Bellman-Ford produce exactly equal float
distance labels. The fixtures exercise unequal fractional weights, a
zero-weight edge and cycle, parallel edges, a self-loop, disconnected vertices,
multiple targets and sources, spatially reordered vertices, and spill-tile
metadata.

Post-pass reconstruction is tested independently of relaxation scheduling.
Every reachable reconstructed path passes endpoint, edge-continuity,
source-termination, exact-tightness, and ordinary-float cost validation.
Unreachable targets return no path. Equal-cost alternatives choose stable edge
IDs deterministically, while a dedicated zero-cycle fixture proves that the
reconstructor backtracks instead of greedily accepting a cyclic predecessor.

All deliberately representable expected distances and path costs are checked
exactly. The four-ULP helper for general-data verification is unit tested, but
is not used to make algorithm or reconstruction decisions.

The `bfnew.graph` correctness test validates deterministic dual sparse-layout
construction and its transpose invariants. This is structural validation, not
a shortest-path or performance result.

The `bfnew.reorder` synthetic layout report measures destination-tile runs in
one eight-edge outgoing row. Stable destination-ID order produced 7 tile runs;
destination-tile grouping produced 4. This confirms the intended grouping on
that fixture only and is not evidence of a runtime or cache improvement.

The `bfnew.tile` synthetic fixture has 12 logical edges: 6 internal and 6
outgoing cross-tile source classifications. The same 6 cross edges have
destination-owned incoming references, and the per-tile halo lists contain 9
deduplicated remote vertex references in total. These are deterministic
structural counts, not performance measurements.

Future performance results will record the exact commands, inputs, seeds, correctness
comparisons, timing distributions, work counters, and memory measurements
required by their implementation phase. Phase 5 reports correctness only and
makes no speed, memory-efficiency, or scalability claim.

## Phase 6 acceptance

Phase 6 passed under the maintainer's CPU-only acceptance policy. A fresh
CPU-only CMake build completed without ROCm and the unchanged five-test suite
passed. The optional HIP probe, repeated profiling workload, profiler script,
and cooperative-barrier benchmark were not compiled or executed on a GPU, so
they carry no GPU correctness or performance claim.

## Phase 12 CPU acceptance

Phase 12 adds a host-only comparison evidence model rather than a measured CPU
or GPU optimization. The focused `bfnew.shootout` test covers the canonical
three-engine configuration catalog, runtime-legality decisions, deterministic
five-dimensional representative selection over at least 1,000 synthetic
metadata rows, interleaved scheduling, correctness-stage gating, timing/counter/
profiler separation, evidence states, quantiles, throughput arithmetic,
long-tail retention, fingerprint mismatch rejection, pending conclusions and
defaults, and byte-stable TSV/JSON output. Its adversarial graph checks remain
tiny and bounded.

Final CPU evidence: the Release build in `build/phase12-cpu` passed all 12/12
bounded tests in 1.10 seconds with
`ctest --test-dir build/phase12-cpu --output-on-failure`. The focused
`bfnew.shootout` binary passed and stated that it collected no GPU timing
evidence. The ASan/UBSan build in `build/phase12-asan` passed that focused test
1/1 in 0.31 seconds.

This CPU evidence does not include the missing real `logicnets_jscl` query
artifact, a representative 1,000-query execution, HIP compilation, GPU
correctness, actual-kernel occupancy, timing, tracing, PMC collection,
long-tail diagnosis, a performance conclusion, or a recommended default. All
seven Phase 12 questions and every default remain pending the deferred measured
GPU campaign in `docs/PHASE12_GPU_VALIDATION.md`.

## Phase 13 CPU acceptance

Phase 13 adds deterministic host planning and workspace accounting, not a CPU
or GPU SSSP performance optimization. The focused plan fixture validates exact
tile-pair indexed vertex/edge estimates, one- and two-source feature retention,
the golden overlap order, stable TSV, permutation independence, the 32/16/8
family, singleton/padded remainders, once-only assignment, and rejection of
changed masks, estimates, identities, duplicates, and omissions.

The focused layout fixture builds retained per-run masks and compact nonzero
descriptors for the same batch. It checks exact source/target offsets, selected
counts, union masks and vertex ranges, touched-ledger reuse, run visit/write
counters, descriptor order and per-union-vertex offsets, a literal
noncontiguous compact tile-bias mapping and global/compact round trips,
CSR/CSC lane-edge totals, and endpoint-by-endpoint admission equivalence. Cold
initialization, warmed touched-ledger reuse, and representation-bound reuse are
checked separately. The workspace fixture compares all four combinations of
full/compact vertex storage and retained/descriptor run storage under an
explicit tiny synthetic budget. Checked byte totals, preparation writes,
selected/wasted lane vertices, allocation reuse, and capacity rejection are
structural measurements of that fixture only.

The provisional bounded-synthetic decision is full graph-sized vertex-major
storage with retained per-run masks. The final tiny-fixture integration values
are:

- device capacity: `65536` bytes;
- resident graph allowance: `4096` bytes;
- explicit reserve: `4096` bytes;
- lane width and slots: `8` lanes and `2` slots;
- selected total/preparation bytes:
  `1012` / `152`;
- selected mapping/run build time:
  `0` / `583` nanoseconds.

The same clean run observed a compact bias-mapping mean of `11` nanoseconds
over 1,024 warmed preparations and a `458`-nanosecond compact-descriptor run
build. Run-image timings cover both CSR and CSC host proof images and are not
device-kernel timing.

The final bounded Release suite passed `14/14` tests in `0.05` seconds. A
separate focused invocation passed the plan and workspace tests `2/2` in
`0.26` seconds. Focused ASan+UBSan validation passed `2/2` in `0.45` seconds.

The Phase 7 partial record gives `V = 28,226,432`. Exact analytical distance
label bytes for widths 8/16/32 are 903,245,824 / 1,806,491,648 /
3,612,983,296 with one slot, and 1,806,491,648 / 3,612,983,296 /
7,225,966,592 with two slots. The illustrative 64 GiB nominal capacity, 8 GiB
resident allowance, 4 GiB reserve, two-slot scenario is not measured free
memory and omits real run/mapping/metadata storage.

No real `.bfqueries` artifact, real batch union/run distribution, full-graph
scan, HIP compiler, GPU allocation, batched kernel, timing, trace, or PMC
evidence is included. Consequently the production workspace choice and every
device-performance conclusion remain deferred as described in
`docs/PHASE13_WORKSPACE_DECISION.md`.

## Phase 14 CPU acceptance

Phase 14 adds a portable semantic/control implementation of overlapping
batched Jacobi pull rather than a CPU performance optimization. Its distance
state is vertex-major with contiguous lanes and two slots. Independent lane
source/target offsets, exact selected tile and CSC run masks, strict float
relaxation, per-lane convergence, final target classification, and all three
controller modes are exercised only on small deterministic graphs.

The focused bounded `bfnew.batched_jacobi_pull` CTest covers widths 1, 8, 16,
and 32; low-prefix validity with padded lanes; one- and multi-source lanes;
immediate, one-change-round, several-round, and unreachable cases; retained
masks and compact descriptors; convergence enabled and disabled; persistent,
chunked, and per-round controls; exact shared-edge, lane-edge, run, utilization,
tail, modeled selected-device reset, and union-inflation counters; and maximum-
round behavior. It verifies that an early-converged lane remains
bitwise equal across both selected-region columns after later batch-wide slot
swaps. Every valid lane is compared with an independently induced bounded
Dijkstra query, and width one is compared bitwise with standalone Jacobi.

Final CPU evidence: the Release `build/phase14-cpu` suite passed all `15/15`
bounded CTests in `1.38` seconds. Its focused Phase 13+14 matrix passed `3/3` in
`0.02` seconds; `bfnew.batched_jacobi_pull` passed `1/1` in `0.01` seconds in
the final full run. The ASan+UBSan `build/phase14-asan` suite passed all
`15/15` in `2.73` seconds, including the Phase 14 CTest `1/1` in `0.20`
seconds.

The reset counter models the writes the selected-range device initializer must
perform; it does not claim that fresh portable vector construction writes only
selected host cells.

These local runs must remain bounded. They do not consume the absent
`logicnets_jscl.padding1.v1.bfqueries` artifact, scan the full recorded graph,
compile HIP, allocate GPU memory, execute the device kernels, or measure
register pressure, occupancy, L2/read traffic, per-query latency, throughput,
or batching benefit. The compiler-uniform edge-load default and explicit-wave-
broadcast experiment have no local performance comparison. All such evidence
is deferred to `docs/PHASE14_GPU_VALIDATION.md`; CPU counters and timings are
not proxies for it.

## Phase 15 CPU acceptance

Phase 15 adds a portable semantic and controller model for overlapping batched
dense chaotic CSR push, not a CPU or GPU performance optimization. Its
distance state is one graph-sized, vertex-major unsigned IEEE-754 word per
configured lane. Sources, targets, selected regions, changed bits, convergence
rounds, target classification, and distance updates remain independent per
valid lane; padded lanes perform no semantic work.

The bounded `bfnew.batched_dense_chaotic_push` CTest is the local Phase 15
acceptance surface. Its fixture matrix covers widths 1, 8, 16, and 32;
retained CSR masks and compact nonzero descriptors; independent and
multi-source lanes; low-prefix padding; no-first-decrease, short, several-scan,
and unreachable cases; cross-lane isolation; per-lane convergence enabled and
disabled; persistent, per-round, and chunked controls for
`K = 2,4,8,16,32`; None/Light/Debug instrumentation separation; exact target
masks and logical-work identities; maximum-round exits; width-one bitwise
agreement with standalone dense push; and an independent bounded Dijkstra
comparison for every valid lane. These are small deterministic checks and do
not search the recorded full graph.

The work report separates considered, visited, and skipped CSR runs;
algorithmic CSR edge-record requests; logical lane-edge relaxations;
atomic-compatible source loads and destination-min attempts; useful updates;
complete scans; convergence tails; and avoided work. In particular,
`csr_edge_loads` is a logical algorithmic request count. It is not a physical
cache-load count, an L2 transaction measurement, or evidence about compiler
load generation. Exact valid-lane, configured-width, wave32, unused-wave,
padded-lane, and edge-wave capacities are arithmetic denominators, not GPU
occupancy. Reset accounting models one selected 32-bit word per admitted
vertex/lane, with source seeds as the zero-valued subset; it does not count
convenient initialization of nonsemantic cells in a fresh host vector.
The shared `high_contention_destinations` result field is a standalone dense-
engine diagnostic and is unavailable for batched Phase 15; zero is not a
measured no-contention result.

Final Release evidence: `build/phase15-cpu` passed all `16/16` bounded CTests
in `0.17` seconds. The focused `bfnew.batched_dense_chaotic_push` CTest passed
`1/1` in `0.01` seconds.

Final ASan+UBSan evidence: `build/phase15-asan` passed all `16/16` bounded
CTests in `1.82` seconds, including `bfnew.batched_dense_chaotic_push` `1/1`
in `0.14` seconds.

No Phase 15 local evidence includes the absent real
`logicnets_jscl.padding1.v1.bfqueries` artifact, a full-graph scan, HIP
compilation, GPU allocation or execution, device correctness, register
pressure, actual occupancy, physical L2 reads or writes, atomic/write stalls,
latency, throughput, batching benefit, or a compiler-uniform versus explicit-
broadcast result. All such evidence is unavailable and deferred to the
combined campaign after all implementation phases, using
`docs/PHASE15_GPU_VALIDATION.md`. Phase 15 itself stops before the separately
implemented Phase 16 batched frontier engine.

## Phase 16 CPU acceptance

Phase 16 adds a portable semantic and controller model for overlapping batched
active-frontier CSR push, not a CPU or GPU performance optimization. Each lane
has independent vertex-major unsigned-float distance words and terminal slices.
The shared scheduling state consists of two vertex queues and two per-vertex
lane-mask arrays, so several query lanes can share one worklist vertex without
sharing labels.

The bounded `bfnew.batched_frontier_push` CTest is the local Phase 16
acceptance surface. Its frozen-tree matrix covers widths 1, 8, 16, and 32;
retained CSR masks and compact nonzero descriptors with equality across the
whole configured `V * W` output image and terminal lane result; independent,
multi-source, and shared-source lanes; low-prefix padding; different frontier
emptying rounds and bounded unreachable targets; cross-lane isolation;
per-lane convergence enabled and disabled; persistent, per-round, and chunked
controls for
`K = 2,4,8,16,32`; shared-destination queue merging; repeated improvements,
high fan-in, rapid expansion, and zero-weight cycles; exact queue/mask/run/
counter invariants; initialization and round overflow; None/Light/Debug
instrumentation separation; clean maximum-round evidence versus error paths
that publish no convergence proof; width-one bitwise agreement with standalone
frontier push; and an independent bounded Dijkstra comparison for every valid
lane. These fixtures are deliberately small and do not search the recorded
full graph.

The exact HIP-off executable/CTest pair is
`bfnew_batched_frontier_push_test` / `bfnew.batched_frontier_push`. The
deferred executable/CTest pair is `bfnew_batched_frontier_push_hip_test` /
`bfnew.batched_frontier_push_gpu`, with a 240-second CTest timeout.

The full public output is hardened across the entire configured
vertex-by-width image: every padded-lane word and every nonselected word in a
valid lane is positive infinity. Portable execution and downloaded correctness
output therefore do not leak stale retained cells. That host/output
normalization is not counted as device reset traffic; the reset counter remains
the exact selected distance words plus both activity masks for union vertices.
The retained/descriptor parity assertion includes all such padded and
nonselected cells, not only each lane's selected region.

The work report distinguishes physical worklist vertices from logical vertex/
lane pairs, shared CSR edge-record requests from lane-edge relaxations, unique
next vertex/lane activations from physical queue claims, and exact queue entries
saved by lane merging. It also retains mask atomics, duplicate suppressions,
queue high-water/overflow, run visits/skips, active lanes over runs, utilization
capacity, lane tails, and modeled selected distance/activity reset traffic.
These are algorithmic counts. They are not physical cache traffic, measured
occupancy, device timing, or evidence of a speedup. For the HIP-facing
vocabulary, `Light` is restricted to aggregate work, sharing, and queue
high-water; `Debug` adds mask atomics, unique activations, claims, merging,
duplicate, and overflow evidence.

The portable engine validates the full queue/activity-mask invariant set at
every bounded round boundary. Its CPU result exposes the final queue and mask
images plus per-round current-size and lane-union traces. The current HIP result
and deferred HIP CTest do not expose a complete per-round ledger; device proof
remains a future or manual Debug diagnostic requirement.

Final post-hardening CPU evidence is HIP-off and bounded:

```text
Release full bounded suite:      17/17 passed in 1.56 seconds
Phase 16 CTest in Release suite: 1/1 passed in 0.02 seconds
ASan+UBSan full bounded suite:   17/17 passed in 3.03 seconds
Phase 16 CTest in sanitized run: 1/1 passed in 0.16 seconds
```

The deferred `bfnew_batched_frontier_push_hip_test` passed strict-warnings host
syntax and fake-HIP `__HIPCC__` syntax, including its device probe. The Phase 16
public headers passed strict host syntax, and both production translation
units, `batched_frontier_workspace.hip.cpp` and
`batched_frontier_push.hip.cpp`, passed strict fake-HIP syntax. These checks did
not invoke a HIP compiler or GPU. No browser, cloud service, real query
artifact, or large/full-graph test was used.

No Phase 16 local evidence includes the absent real
`logicnets_jscl.padding1.v1.bfqueries` artifact, a full-graph scan, HIP
compilation, GPU allocation or execution, device correctness, actual kernel
occupancy, physical L2 or memory traffic, latency, throughput, batching
benefit, or comparison with the earlier batched engines. All such evidence is
unavailable and deferred to the combined campaign after all implementation
phases, using `docs/PHASE16_GPU_VALIDATION.md`.

## Phase 17 CPU acceptance

Phase 17 adds a portable all-query orchestration and evidence model above the
three existing batched engines. It is not a CPU performance recommendation and
does not establish GPU behavior. The controller consumes clean per-batch
reached/miss status, accepts reached lanes, expands all misses after the pass,
deterministically replans them, and restarts each query from its original
source set with a new generation.

The exact HIP-off executable/CTest pair is
`bfnew_batched_expansion_test` / `bfnew.batched_expansion`. Its bounded matrix
covers:

- one-ring, fixed-two-ring, doubled-margin, and hybrid-two-small-step schedules
  across portable batched Jacobi, dense, and frontier execution;
- five simultaneous initial misses beside one reached lane, plus a separate
  one-miss retry;
- northern/southern dissimilar expanded regions, a fourth-ring cross-tile
  intermediate, spill-target adjacency, multi-source restart, several
  generations, full fallback, explicit expansion limit, stalled region, and
  globally unreachable full-region termination;
- exact original terminal/source maps, deterministic input reorder and retry
  trace, nonzero initial generation, valid `UINT32_MAX` QueryId, and contained
  per-query generation/count overflow (aggregate telemetry overflow remains a
  fail-closed campaign error);
- errors and maximum-round exits that expand no query; rejection of incomplete
  clean partitions, classification on nonclean status, malformed label image,
  invalid work state, and unavailable work carrying numeric values;
- exact aggregate, retry, and failed-batch shared/logical work, expansion
  histogram, utilization, region-size, required execution-configuration and
  comparison fingerprints, invalid-campaign and exact-score-tie rejection, and
  bounded host-throughput accounting;
- retained run-mask versus compact-descriptor terminal parity; and
- bitwise Dijkstra agreement on every terminal query's final admitted induced
  region, including no missed target reported as success.

Final bounded evidence after the strengthened test matrix is:

```text
Release full bounded suite:      18/18 passed in 1.70 seconds
Phase 17 CTest in Release suite: 1/1 passed in 0.01 seconds
ASan+UBSan full bounded suite:   18/18 passed in 3.32 seconds
Phase 17 CTest in sanitized run: 1/1 passed in 0.19 seconds
```

The focused gate also passed strict C++20 warnings, and the production core
source passed the same strict syntax check. The deferred HIP test passed strict
host and fake-HIP syntax; that source covers all executor bindings and
schedules, compact/evidence transfer policy, reuse, retry/fallback identity,
one-miss compaction, persistent compact control, and maximum-round
no-expansion. None of these source checks invokes a HIP compiler or GPU.

No local result includes the absent
`logicnets_jscl.padding1.v1.bfqueries` artifact, a large/full graph, HIP
compilation, device execution, compact 48-byte transfer traces, device
correctness, occupancy, physical memory traffic, latency, real replanning
cost, expansion distributions, all-query throughput, or a schedule
comparison. Those values remain unavailable rather than zero. No schedule is
selected from the tiny synthetic fixture. The combined-campaign procedure is
`docs/PHASE17_GPU_VALIDATION.md`.

Phase 17 stops before compact target/path output and reconstruction
integration. Phase 18 now supplies that separate post-relaxation layer.

## Phase 18 CPU acceptance

Phase 18 adds compact target summaries, deterministic post-relaxation path
reconstruction, generation-bound payload retention, exact compact validation,
and checked result-transfer accounting. It does not change any relaxation
engine and is not a CPU or GPU performance recommendation.

The exact HIP-off executable/CTest pair is `bfnew_compact_paths_test` /
`bfnew.compact_paths`. Its bounded fixture covers:

- a fixed 28-byte per-target summary with explicit reached,
  selected-source-valid, path-length, and reconstruction fields;
- complete path-sized vertex, actual final distance-label, and stable-edge-ID
  arrays, with no graph-sized matrix in a compact result;
- fractional and zero-weight paths, equal-cost alternatives and parallel edges
  with stable-ID choice, and a lower-ID zero-cycle branch that requires true
  backtracking;
- target-is-source, multiple canonical sources, and duplicate original target
  terminals with their terminal-to-target map preserved;
- a bounded route that remains correct while unbounded Dijkstra finds a
  cheaper route outside the selected tiles;
- later-generation long cross-tile and spill paths, multi-source restart, and
  full-region unreachable results through all three portable batched engines;
- one identity-bound payload for every clean valid lane, discard of retry-miss
  generations only after every target is classified complete or unreachable
  and at least one is unreachable, retention of terminal full-region complete/
  unreachable summaries, normalization of all-complete misses,
  reconstruction-failure misses, and reached/incomplete payloads to engine
  failure, and release of portable graph-sized diagnostic images after compact
  extraction, plus fail-closed rejection of wrong-generation, duplicate,
  padded-lane, and off-by-one transfer-accounting payloads;
- widths 1, 8, 16, and 32 for all three portable engines, including explicit
  padded-lane exclusion;
- checked summary/vertex/path-label/edge-ID serialization-byte accounting;
- a checked final-serialization model using 48-byte status records, 28-byte
  target summaries, checked-u32 device vertices/edge IDs, float labels, and
  every reconstruction-status partition, explicitly distinguished from
  physical D2H evidence and discarded retry payloads, with fail-closed
  reach/distance/source/length/arena matrix mutations for noncomplete statuses;
- separate retry-capable compact payload/status/error subtotal, controller-
  poll count/bytes, and overall device-transfer fields, unavailable and zero
  in the portable run plus bounded malformed/aggregate tests for supplied
  device evidence; persistent control requires zero poll traffic, while
  ordinary/chunked poll bytes remain outside the compact subtotal;
- a fail-closed stage ledger covering measured and unavailable evidence,
  separate geometric-expansion and controller/orchestration intervals, the
  exact pre-grown `warm_all_query` sum when every named host stage is measured,
  the valid separately measured warm enclosure when asynchronous execution
  leaves named host stages unavailable rather than measured-zero, an
  independent first capacity-growing `cold_execution`, and the exact artifact-
  load + upload + cold-execution `cold_pipeline` sum, with no cold/warm
  equality; plus malformed enum/value combinations, sum mismatch, and overflow
  without inventing device-event aggregates;
- the assembled host no-congestion pipeline, including payload consumption,
  release of every graph-sized image, checked final serialization, and explicit
  host-wall versus unavailable device timing/transfer evidence;
- input-order-independent SplitMix64 QueryId sampling with a recorded seed,
  absolute and nearest-rank P50/P95/P99/maximum metrics, inclusion of complete
  targets from mixed terminal results, the expected bounded-versus-unbounded
  cost inflation for a cheaper excluded route, zero/zero unit ratios, and a
  positive-infinite ratio with finite absolute inflation for an excluded global
  zero-cost detour;
- independent source/target, shape, simple-path, selected-region, stable-edge,
  exact compact-label tightness, and reported-cost validation;
- diagnostic comparison against the full terminal label projection without
  making that projection part of production output; and
- fail-closed rejection of changed QueryId/generation/terminal maps, every
  mismatched reached/unreachable/failure disposition-status combination,
  unknown disposition values, reordered or changed target identity, unknown
  reach/reconstruction enums, invalid
  source-valid flags, selected-source mismatch, selected-region escape,
  repeated vertices, out-of-range or discontinuous edge IDs, malformed lengths,
  wrong cost, non-tight labels, missing reached images, and label-free reached
  failures.

All deliberately representable compact distances, compact path labels, and
reported costs use bitwise exact comparisons. The selected-region path is the
correctness result even when it is more expensive than the sampled unbounded
path.

Final bounded evidence is recorded after the clean integrated run:

```text
HIP-off Release: 19/19 passed in 2.02 seconds
  bfnew.compact_paths: 1/1 passed in 0.10 seconds
HIP-off ASan+UBSan: 19/19 passed in 2.82 seconds
  bfnew.compact_paths: 1/1 passed in 0.16 seconds
Strict host source/public-header syntax: warning-clean
Fake-__HIPCC__ production, CLI, and deferred-test syntax: warning-clean
```

The strict source checks are separate from the CTest results. Neither is HIP
compilation or device execution. No local result uses
the absent `logicnets_jscl.padding1.v1.bfqueries` artifact, a large/full graph,
GPU transfer trace, device correctness, occupancy, physical traffic, latency,
throughput, representative stage-time distributions, device-event timing, or
representative-corpus path-inflation distribution. Those values remain
unavailable rather than zero. The bounded synthetic ledger/sample tests are
contract evidence, not performance evidence. The deferred procedure is
`docs/PHASE18_GPU_VALIDATION.md`.

Phase 18 stops before congestion/resource-conflict logic and adaptive engine
hybridization. Phase 19 now supplies the separate final audit.

## Phase 19 CPU acceptance

Phase 19 adds a fail-closed evidence audit and no algorithm. The exact HIP-off
executable/CTest pair is `bfnew_final_audit_test` / `bfnew.final_audit`.
Its bounded matrix requires:

- all 56 feature, 27 comparison, 10 profiler-metric, 7 question, and 15 blocker
  identities exactly once in canonical order;
- validator recognition of all four feature-classification encodings, while
  the canonical local report partitions them 36/0/12/8 and therefore has
  exactly zero class-2 rows;
- strict separation of portable correctness from optional HIP/device evidence;
- every engine-specific control, ordinary-`K`, and persistent-grid comparison,
  plus width, planner, expansion, latency, throughput, end-to-end, rate,
  quality, memory, copy/synchronization, and profiler rows;
- the seven explicit Phase 19 questions, distinct from Phase 12;
- exact local evidence blockers and an absent production recommendation;
- deterministic serialization and byte-stable validated round trips, with
  nonzero, strictly increasing supporting configuration references; and
- fail-closed rejection of missing, duplicate, reordered, or unknown IDs;
  invalid enum values; unavailable rows carrying numeric/configuration/
  provenance data; inconsistent evidence counts/fingerprints; premature
  performance-profiled features, measured comparisons/questions, hybrid
  decisions, or recommendations; malformed production configurations or an
  unresolved unique-winner gate; mismatched profiler/best-engine identities;
  nonfinite or signed-negative-zero numeric rows; duplicate support references;
  and malformed or trailing serialized records.

The generation inputs `logicnets_jscl_unrouted.phys` and
`logicnets_jscl.netlist` are present under the configured FPGA24 root. The
required generated `logicnets_jscl.padding1.v1.bfqueries` artifact is not.
There is also no local HIP compilation/device run, representative corpus
timing, accepted profiler evidence, matched CPU baseline, or normalized Phase
18 tail provenance. The accepted local audit must therefore contain no class-2
SSSP feature, measured comparison/question, configuration recommendation, or
adaptive-hybrid decision.

Final evidence from the clean integrated run:

```text
HIP-off Release: 20/20 passed in 2.12 seconds
  bfnew.final_audit: 1/1 passed in 0.01 seconds
HIP-off ASan+UBSan: 20/20 passed in 3.63 seconds
  bfnew.final_audit: 1/1 passed in 0.04 seconds
Strict Phase 19 source/public-header/test syntax: passed warning-clean with
  C++20, -Wall, -Wextra, -Wpedantic, -Wconversion, -Wsign-conversion,
  -Wshadow, and -Werror
```

These results validate the audit contract and its refusal to overclaim. They
do not add GPU correctness, timing, profiler, or production evidence. The final
comparison and exact missing gates are in `docs/PHASE19_FINAL_AUDIT.md`.
