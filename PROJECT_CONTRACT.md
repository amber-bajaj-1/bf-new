# Project Contract

## Purpose and current scope

`bfnew` is an independently implemented prototype for weighted shortest paths
on directed graphs. Development is divided into small, independently accepted
phases.
Phase 0 established this contract and the C++20 CMake/CTest scaffold. Phase 1
added validated record-level graph inputs and deterministic test fixtures.
Phase 2 added immutable deterministic outgoing CSR and incoming CSC layouts.
Phase 3 added deterministic spatial vertex permutation and tile-grouped sparse
layout ordering. Phase 4 added the uniform-grid partitioner and immutable tile
directory metadata. Phase 5 added CPU-only Dijkstra and synchronous weighted
Bellman-Ford correctness solvers plus deterministic post-relaxation path
reconstruction. The replacement GPU campaign begins at Phase 6 with optional
HIP build support, a capability probe, a profiling gate, and a cooperative
barrier microbenchmark. Phase 7 adds the independent FPGA Interchange workload
bridge, versioned graph/query artifacts, and the first solver-facing
`RouteQuery`; it still adds no GPU SSSP engine.
Phase 8 adds the shared device-foundation contract: deterministic maximal
CSR/CSC tile runs, a checked 32-bit relaxation-hot graph staging layout,
fixed-width engine/controller/status records, and retained workspace planning.
When HIP is explicitly enabled, the same phase also provides stream/event and
allocation RAII, immutable one-upload resident graph ownership, retained
query/workspace buffers, asynchronous transfers, and event/wall timers. The
host graph continues to use 64-bit edge offsets. Phase 8 still implements no
SSSP kernel and makes no GPU-performance claim.

Phase 9 adds only the standalone synchronous Jacobi CSC-pull engine. A
portable host implementation models its relaxation and all three controller
protocols so the semantics can be exercised on a CPU-only system. A separately
gated HIP implementation contains the ordinary round/advance kernels and one
persistent cooperative kernel. The HIP source passed only a host-side
fake-declaration syntax check; no HIP compiler or GPU executed it in the local
Phase 9 pass, so it is not GPU-validation or performance evidence.

Phase 10 adds only the standalone dense chaotic CSR-push engine. A portable
host implementation proves the restricted unsigned-float atomic-min domain,
full-edge-scan semantics, and all three controller protocols. A separately
gated HIP source implements CAS-compatible distance loads and updates,
ordinary round/advance pairs, and one persistent cooperative query kernel.
It has not been compiled by a HIP compiler or run on a GPU locally.

Phase 11 adds only the standalone active-frontier/worklist CSR-push engine. A
portable host implementation exercises its two bounded queues, per-generation
deduplication, strict atomic-min model, explicit overflow behavior, and all
three controller protocols. A separately gated HIP source implements the same
queue protocol with device atomics, ordinary round/advance pairs, and one
persistent cooperative query kernel. The HIP source has not been compiled by
a HIP compiler or run on a GPU locally.

Phase 12 adds the controlled single-query shootout evidence layer without
adding an engine or selecting an adaptive combination. It provides a canonical
three-engine tuning catalog, runtime legality records, fingerprinted and
stratified representative-query manifests, deterministic interleaved schedules,
strictly separated correctness/timing/counter/profiler records, completeness
gates, deterministic TSV/JSON serialization, distribution and long-tail
summaries, and explicit unavailable/not-applicable/measured evidence states.
This host infrastructure is CPU-tested. The real `logicnets_jscl` query
artifact is not present, and no HIP compiler, GPU, representative 1,000-query
campaign, timing run, profiler run, performance conclusion, or recommended
default has validated it.

Phase 13 adds the host-only deterministic overlapping-batch planner, exact
batch-device descriptions, and checked workspace-strategy comparison. It
produces canonical plans for widths 32, 16, and 8; retains every query exactly
once, including padded singleton remainders; and materializes exact per-tile
and per-run lane admission without implementing a batched SSSP kernel. The
workspace choice is provisionally full graph-sized vertex-major lane storage
with retained per-run masks, based only on bounded synthetic measurements and
an analytical recorded-vertex-count capacity check. The missing real
`logicnets_jscl` `.bfqueries` artifact, real union/run distributions, runtime
free-device-memory observation, HIP compilation, and GPU execution prevent a
production workspace decision or performance claim.

Phase 14 adds overlapping batched Jacobi pull for widths 1, 8, 16, and 32. A
portable host implementation exercises independent lane sources, endpoint-
exact CSC run admission, per-lane convergence, two-buffer parity, target
classification, work accounting, and all three controller protocols on bounded
fixtures. The separately gated HIP implementation has a batch-specific
workspace and engine surface, wave32 destination ownership, retained CSC run
capacity with selected masks materialized on-device during initialization,
ordinary and persistent control paths, and selectable compiler-uniform
or explicit-wave-broadcast edge loads. The compiler-uniform path remains the
default; the broadcast path is an unselected experiment until a target-device
comparison exists. No HIP compiler or GPU executes during local acceptance,
so Phase 14 makes no device-correctness, throughput, occupancy, register, L2,
or optimization claim.

Phase 15 adds overlapping batched dense chaotic push for widths 1, 8, 16, and
32. Its portable host implementation uses separate vertex-major atomic words
per lane, endpoint-exact outgoing CSR run admission, complete in-place scans,
per-lane convergence, post-convergence target classification, and all three
controller protocols. A separately gated HIP source exposes
`hip::BatchedDenseChaoticPushEngine`, `ReusableBatchedDenseWorkspace`, and
selectable compiler-uniform or explicit-wave-broadcast edge-field paths over
the provisional full-vertex/retained-CSR layout. Those HIP translation units
have not been compiled or executed locally. Device correctness, register
pressure, occupancy, L2 traffic, write stalls, latency, throughput, batching
benefit, and the uniform/broadcast decision are unavailable, not zero. The
real `logicnets_jscl` query artifact is also absent. Phase 16 batched frontier
push adds a third, separately implemented overlapping batched engine for widths
1, 8, 16, and 32. It retains one independent distance word per vertex/lane,
two vertex queues, and two per-vertex lane-mask arrays. A queue entry names one
vertex active for at least one lane; the zero-to-nonzero transition of the next
mask deduplicates that vertex. Its portable model covers the two run
representations, both convergence settings, all three controls, queue overflow,
target classification, and exact bounded-oracle comparisons. The separately
gated `hip::BatchedFrontierPushEngine` and
`ReusableBatchedFrontierWorkspace` expose the same queue/mask protocol. No HIP
compiler or GPU executes during local Phase 16 acceptance, so
device correctness, occupancy, physical memory traffic, latency, throughput,
and comparisons with the other batched engines remain unavailable.

Phase 17 adds deterministic all-query execution above those three batch
engines. It accepts reached lanes, expands only clean bounding-region misses,
replans failures by QueryId, increments generation, and restarts from original
sources under one of four explicit schedules. It provides one final full-
region fallback or explicit terminal failure, a complete retry/region/work
evidence record, and no guessed schedule default. The HIP-gated adapter copies
one compact status per completed batch and reuses the existing engine reset and
workspace paths. No real query artifact, HIP compiler, GPU, corpus run, or
schedule comparison is part of local acceptance. A separately gated
`bfnew_gpu_batched_expansion` source now provides the deferred all-query
artifact-to-report campaign boundary when both HIP and FPGAIF are enabled; it
has not been compiled or executed locally.

Phase 18 adds compact target/path production and the no-congestion end-to-end
result boundary. A fixed 28-byte target summary reports the canonical target,
distance, reached and reconstruction statuses, edge-count path length, and an
explicit selected-source-valid word plus source ID. Complete paths carry only
their global vertex IDs, actual final path-label values, and stable logical
edge IDs. The portable implementation reconstructs from the terminal distance
projection and incoming CSC with stable-edge ordering, path-local cycle
detection, and real backtracking. The ordinary validator uses the compact
per-path labels for exact tightness, so production validation needs no full
lane matrix. A separate full-image validator remains available only for
bounded correctness sampling. Phase 18 adds no congestion, resource-conflict,
exit-certificate, or adaptive-engine logic. Optional HIP reconstruction and
the artifact campaign have not been compiled or executed locally.

Phase 19 adds only the final audit and evidence-based recommendation boundary.
It inventories every standalone/batched/control/expansion/reconstruction and
measurement surface, assigns one of four explicit evidence classifications,
and represents each required comparison and project-level question without
inventing evidence. The local snapshot has no performance-profiled SSSP
feature, no measured performance conclusion, and no production configuration.
Its correct result is insufficient representative evidence and no
recommendation. Phase 19 adds no algorithm and is the terminal phase.

Phase 8 HIP query leases bind preparation, helper-mediated result transfers,
and retirement to one stream. Transfer calls are stream ordered, but a
pageable host endpoint is not evidence of host/device overlap; guaranteed host
overlap requires HIP page-locked memory. Workspace reports distinguish device
bytes, pinned-host staging bytes, and their combined requirement.

Phase 9 uses the retained engine scratch as exactly two graph-sized float
distance columns. GPU-event timing, where later executed, is a device-timeline
span from initialization through terminal status materialization; final result
transfer completion and lease retirement share one stream boundary and remain
included in host wall time. In
host-poll modes the event span includes stream-idle gaps while the host checks
and re-enqueues work, so it is not summed kernel-active time. Neither the
portable host timing nor an unexecuted timer implementation is GPU-performance
evidence.

Phase 10 uses exactly one graph-sized 32-bit distance-word array. It allocates
no predecessor, second distance column, frontier, queue, or worklist storage.
Its host-poll GPU-event span has the same idle-gap limitation as Phase 9.

Phase 11 retains one graph-sized 32-bit distance-word array, two bounded
32-bit vertex queues, and one graph-sized 64-bit enqueue-generation array.
The default queue capacity is one entry per vertex. A smaller explicit
capacity is a validation seam for the required overflow stop; it is not a
silent truncation mode. Its host-poll GPU-event span has the same idle-gap
limitation as Phases 9 and 10.

Phase 12 adds no relaxation scratch and changes none of those engine memory
contracts. Its timing-record contract separates query preparation, the SSSP
device-timeline span, result transfer, and end-to-end host wall time. A final
timing sample uses `InstrumentationLevel::none` and a status-only result path;
graph-sized distance downloads belong only to the separate correctness stage.
The current Phase 12 implementation validates and serializes that boundary but
does not provide or claim a locally executed HIP campaign.

Phase 13 describes, but does not allocate, future batched distance storage.
The checked comparison covers full graph-sized vertex-major lanes and compact
union-tile lanes, each paired with either a retained dense per-run mask array
or sorted compact nonzero `(run_id,lane_mask)` descriptors. Estimates separate
distance bytes, tile mapping, run storage, descriptor offsets, batch metadata,
reset/write traffic, selected versus wasted lane vertices, allocation reuse,
and maximum concurrent workspace count under an explicit budget. No estimate
is a device allocation or a bandwidth/timing measurement.

Phase 14 uses the full graph-sized, vertex-major layout
with contiguous query lanes and two distance slots. Only selected lane/vertex
cells are initialized and read semantically; padded and nonselected cells are
not correctness state. Its dedicated retained workspace adds flattened
terminal offsets, selected ranges, lane-convergence rounds, a device-only dense
CSC mask image, controller/status records, optional instrumentation, and checked
geometrically retained distance capacity. This implementation choice does not
promote the Phase 13 bounded-synthetic vertex-layout decision to a measured
production default.

Phase 15 consumes the same provisional full graph-sized, vertex-major layout
with one 32-bit atomic distance word per vertex/query lane and exact retained
CSR run masks. Its batch-specific retained workspace owns flattened terminals,
selected ranges, lane-convergence rounds, controller/status/statistics records,
and geometrically retained one-slot scratch. Initialization writes only
selected range/lane words; padded and nonselected cells are nonsemantic. The
portable compact-descriptor path remains equivalence evidence, not a claim that
the first HIP path implements or benefits from compact storage. This choice is
still provisional because the real query artifact, runtime free memory, HIP
compilation, and device measurements are absent.

Phase 16 also consumes one full graph-sized vertex-major distance word per
vertex/lane and the retained CSR run-mask image. Frontier scratch additionally
needs two bounded 32-bit vertex queues and two graph-sized 32-bit per-vertex
lane-mask arrays. A zero requested queue capacity means one vertex entry per
graph vertex; a smaller explicit capacity exists only to validate overflow.
For vertex count `V`, width `W`, and queue capacity `Q`, exact scratch is
`4*V*W + 8*Q + 8*V` bytes; default `Q = V` gives `4*V*W + 16*V`. The generic
Phase 13 one-slot estimate did not include either queue or either activity-mask
array, so it is not a complete Phase 16 allocation estimate. The dedicated
Phase 16 scratch calculation checks those additive bytes before any allocation.
Full-vertex labels plus retained CSR masks remain provisional because the real
query artifact, runtime free memory, HIP compilation, and device measurements
are absent.

## Workspace ownership

- `/Users/amber_bajaj/Desktop/RIPS/bf-new` is the only writable project
  directory.
- `/Users/amber_bajaj/Desktop/RIPS/rips2026-amd-routing` is read-only
  historical background. It may be inspected only to understand externally
  observable behavior, documented artifact formats, FPGAIF ingestion and
  device-graph semantics, terminal mapping, and compact path-processing
  concepts. Historical source text, tests, implementation fragments,
  constants, data tables, controller or queue logic, synchronization logic,
  workspace implementation, and GPU kernels must not be copied, pasted,
  translated, mechanically transformed, transplanted, imported, or linked.
- A narrow adapter may be independently implemented from documented artifacts,
  FPGA Interchange schemas, and validated observable behavior. It must have
  focused validation against the current `WeightedGraph`, `PartitionedGraph`,
  and query semantics and must compile without a historical source dependency.
- `/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest` remains read-only.
  It is an external input and schema/API source in phases that explicitly use
  it, and it must never be edited.
- FPGA Interchange schemas are expected under the read-only root
  `/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest/fpga-interchange-schema/interchange`.
  Both read-only roots are configurable CMake cache paths and are not
  hard-coded library dependencies.
- Generated schema sources must be written below `bf-new/build`, never beside
  the source schemas.
- Every generated report, cache, converted graph, and temporary project
  artifact must be placed below `bf-new/build`, `bf-new/out`, or an operating
  system temporary directory.
- Each phase must end by verifying that neither read-only repository was
  modified by the work.

## Graph and algorithm requirements

- The graph is directed. A bidirectional connection is represented by two
  distinct directed edge records.
- Edge weights are arbitrary finite, nonnegative IEEE-754 `float` values.
  Ingestion rejects NaN, infinity, and values strictly below zero. Accepted
  signed zero is normalized to `+0.0f` before sorting, ID assignment,
  validation, or storage.
- Stored weights, distance labels, and relaxation candidates use `float`.
  Relaxation uses ordinary float addition followed by strict `<`, without an
  algorithmic tolerance. Deliberately representable test values are compared
  exactly; general-data verification uses a small, documented ULP-aware
  tolerance.
- There are no negative edges or negative cycles.
- BFS, bitset BFS, unit-weight special cases, and paths that assume every
  weight is one are forbidden. Tests must include unequal and fractional
  positive weights, zero-weight edges, parallel edges, and disconnected
  vertices.
- Dijkstra may exist only as a CPU correctness oracle. The prototype under
  test belongs to the weighted min-plus/Bellman-Ford family. Both use float
  candidate arithmetic.
- The hot relaxation workspace stores distances, not predecessors. Requested
  paths are reconstructed afterward from final distances and graph edges.
  Reconstruction must handle zero-weight cycles using deterministic tight-edge
  traversal, stable edge-ID ordering, path-local cycle detection, and real
  backtracking, or an equivalently deterministic acyclic post-pass parent
  forest.
- Every completed algorithmic phase must compare results with the CPU
  Dijkstra oracle.
- Graph construction, ordering, region growth, batching, reports, and tests
  must be deterministic.
- Edge costs are frozen for the complete duration of one query or overlapping
  query batch. Congestion costs and occupancy/historical congestion updates are
  explicitly deferred; Phase 6 adds no congestion state or placeholder.
- Multi-source routing queries seed every valid source with zero. Future
  workload ingestion excludes driverless nets and `GLOBAL_USEDNET` from route
  queries.

## Identity, provenance, and query requirements

- Strong integer types or wrappers are required for vertex, edge, tile, and
  query IDs. Vertex, tile, and query IDs should be 32-bit; edge counts and
  offsets should be 64-bit. Every narrowing conversion must be validated.
- Logical edge IDs are independent of CSR and CSC array positions. Before
  spatial permutation, they are assigned from a canonical ordering based on
  original source, original destination, canonical float weight bits,
  physical provenance, and parallel rank.
- Physical provenance is a lexicographically comparable 16-byte POD key with
  32-bit `domain`, 32-bit `kind_and_flags`, and 64-bit `source_record` fields.
  Numeric domain and kind constants are centralized and serialized
  explicitly. Pointers, strings, process-dependent hashes, and unordered
  iteration order are not provenance.
- Synthetic edges provide deterministic numeric source records. Distinct
  parallel edges remain distinct. Deterministic input ordinal is only a final
  tie-breaker when the source guarantees deterministic order and otherwise
  identical records cannot be distinguished.
- Every route query has at least one valid source and one valid target.
  Solver-facing terminal sets are sorted by vertex ID and deduplicated after
  range validation. Mappings from original terminal positions to canonical
  IDs are preserved.
- Source-to-all-vertices SSSP, if later required, must use a distinct API; an
  empty target list has no special meaning.

## Spatial-region requirements

- The first spatial partitioner is a uniform grid. A quadtree is deferred
  unless measured tile distributions justify evaluating one.
- The initial query region is the padded bounding rectangle containing every
  located source and target tile. Padding is measured in geometric tile rings.
- One geometric ring is the 8-neighbor, Chebyshev-distance-one ring around the
  selected region.
- Long cross-tile and spill-tile edges remain represented and participate only
  when both endpoint tiles are selected for the query.
- Vertices without valid coordinates belong to an explicit spill tile and are
  never dropped. The spill tile participates in adjacency through actual
  incoming and outgoing graph edges. It is initially admitted when it contains
  a source or target and otherwise enters through normal region-selection and
  expansion rules.
- A halo is metadata for a remote endpoint adjacent to an owned vertex; it
  never owns a duplicate distance value.
- Correctness is equality with Dijkstra on the induced subgraph of the selected
  tiles. Global shortest-path optimality outside that selected region is not
  required, and a reached target is accepted without an exit-edge certificate.
- If a target is unreachable, the lane is recorded as a bounding-box miss. The
  current batch finishes, all failed queries are collected once, expanded,
  deterministically replanned into overlapping batches, and restarted from
  their original source sets. Per-round expansion and label retention across
  expansion are deferred.
- A configurable, measured expansion limit may end in a full-region fallback
  so every routable query has a completion path. Algorithm/controller errors
  are not bounding-box misses.

## Performance and memory requirements

- Device-memory capacity is considered ample based on the supplied
  approximately 8 GB of 64 GB observation. Performance-critical CSR and CSC
  fields may be duplicated to avoid indirection.
- That capacity observation is not evidence about bandwidth, latency, cache
  behavior, or atomic contention. No bottleneck or speedup may be claimed
  without measurements.
- Required counters include examined edges, useful relaxations, frontier
  sizes, batch-lane utilization, host checks, dispatches, and expansion count.
- Prefer structure-of-arrays storage, contiguous lane values, fixed-width IDs,
  stable edge IDs, and trivially copyable device-facing views.
- HIP support is optional. A default CPU-only configuration must build without
  ROCm; explicitly requesting unavailable HIP support must fail clearly.
- The validated AUP target is `gfx1151`: 40 architectural CUs, 20 HIP
  multiprocessors/WGPs, wave size 32, cooperative launch supported and executed,
  ROCm 7.13.0, ROCprofiler-SDK 1.3.0, and no exposed PC sampling. The measured
  trivial-probe ceiling was eight blocks per WGP (160 total). Real kernels must
  query their own occupancy and must not hard-code that ceiling.
- Wave32 is the principal future batch width. Width 64 is a separate two-wave
  experiment, not a default assumption.

## Replacement GPU campaign

The immediate algorithmic objective after the Phase 6 gate is to implement and
compare three standalone, uncombined SSSP engines:

1. synchronous Jacobi min-plus CSC pull;
2. dense in-place chaotic CSR push; and
3. active-frontier/worklist CSR push.

Each engine independently supports persistent cooperative control, chunked
host polling, and per-round host polling through one explicit device-resident
controller protocol. Phase 14 adds overlapping multi-query batching for
Jacobi pull only; batching for the other engines remains separate later work.
Adaptive push/pull switching, disjoint-region batching, target-based early
stopping without a separate proof, and congestion state are outside this
campaign.

## Phase 9 standalone Jacobi contract

- Jacobi owns each admitted destination once and computes its complete next
  label from the immutable preceding label column and admitted incoming CSC
  edges. It always includes the unchanged preceding label as a candidate,
  writes through a distinct next column, uses ordinary float addition and
  strict `<`, and never performs an atomic distance update.
- All valid sources are initialized to exact zero in both distance columns;
  other labels begin at positive infinity. Targets do not terminate
  relaxation early.
- Destination admission is checked from its owner-tile mask. Incoming edges
  are consumed through immutable CSC source-tile runs. The endpoint-tile lane
  intersection is materialized once per run, a zero-mask run is skipped in
  full, and no per-edge tile lookup is part of relaxation.
- Each executed round writes every admitted destination. A block reduces its
  local change observations before at most one global changed-flag update;
  this controller atomic is not a distance atomic. The controller increments
  the completed-round count and swaps the read/write slot identities only
  after a complete round. Normal convergence therefore leaves the two columns
  bitwise identical and makes the final slot explicit in status.
- Ordinary execution uses exactly one engine-round kernel followed by one
  controller-advance kernel. Per-round polling copies and synchronizes once per
  pair. Chunked polling enqueues `K` ordered pairs for `K = 2,4,8,16,32`, then
  performs one controller copy and synchronization; pairs queued after `done`
  cheaply no-op and cannot change actual-round parity. HIP work statistics
  additionally count the initial controller H2D and terminal controller D2H
  records; portable semantic accounting counts only host observations.
- Persistent execution is one cooperative query kernel. It performs
  initialization, grid-stride destination rounds, changed-state reset,
  controller advancement, and final status on the device. One designated
  thread owns global controller transitions, and every workgroup reaches the
  grid barriers surrounding round work and advancement, including the final
  round. No convergence copy or host synchronization occurs inside that
  kernel.
- A persistent launch derives its legal grid from
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` on the real Jacobi kernel and
  selected block/shared-memory configuration. The Phase 6 probe ceiling is
  never reused as a Jacobi limit. Occupancy-derived and explicitly smaller
  legal grids remain independently selectable for later measurement.
- Engine construction compares a deterministic two-word fingerprint of every
  checked 32-bit hot graph/run component with the resident upload. It rejects
  a same-shaped allocation containing different content. This is an accidental
  identity guard, not a cryptographic integrity or authenticity mechanism.
- A converged target set produces either reached or bounding-box-miss lane
  state. Maximum-round exhaustion, invalid controller state, and device errors
  are not bounding-box misses. The result retains the exact Jacobi engine,
  control mode, completed rounds, final distance slot, counters, and timing
  fields.

## Phase 10 standalone dense chaotic-push contract

- Dense push stores one mutable distance per vertex and traverses admitted
  outgoing CSR edges. Every executing round scans the complete admitted edge
  set; it has no frontier, worklist, predecessor, or target-based early stop.
- Each admitted edge atomically loads the current source distance, adds its
  nonnegative weight, and applies a strict atomic minimum to the destination.
  The implementation is asynchronous/chaotic: ordering within a round is not
  a deterministic serial relaxation order, although the converged result must
  be deterministic and agree with bounded Dijkstra.
- Distance words are unsigned IEEE-754 encodings restricted to finite
  nonnegative values, canonical positive zero, and positive infinity. In this
  domain unsigned integer order equals float order. Negative zero, negative
  values, NaNs, and unrepresentable bit patterns are rejected. Ordinary
  distance reads may not race atomic writes; the device load uses the same CAS
  atomic domain as the minimum update.
- A source vertex owns each outgoing CSR row. Source/destination tile
  admission is materialized once per immutable CSR run; a zero-mask run is
  skipped in full, and relaxation performs no per-edge tile lookup. The
  complete-scan no-change result is the only normal convergence condition.
- One controller transition owns changed-mask consumption, executed-round
  accounting, convergence, maximum-round exhaustion, and errors. Dense push
  never swaps distance slots: both slot fields and final status remain zero.
  A block may publish at most one changed flag after reducing local success.
- Ordinary execution uses one initialization kernel and ordered
  `(dense_round_kernel, dense_advance_kernel)` pairs. Per-round control polls
  after one pair. Chunked control submits exactly `K` pairs for
  `K = 2,4,8,16,32`; pairs already queued after device `done` are no-ops.
  Persistent control is exactly one cooperative query kernel containing
  initialization, run-mask construction, scans, controller transitions, and
  terminal status, with uniform grid barriers around every transition.
- Persistent grid legality is derived from occupancy of the real dense kernel
  and its chosen block/shared-memory configuration. The Phase 6 probe ceiling
  is not a dense-kernel limit.
- Light instrumentation records complete scans, admitted edges, decreases,
  and active vertices. Debug additionally records atomic attempts, successful
  atomic updates, run-mask work, block-level changed publications, and unique
  admitted destinations with at least two incoming edges. Instrumentation may
  add counter atomics but must not change labels.
- Bounding-box miss is published only after algorithmic convergence with an
  unreachable requested target. Maximum-round exhaustion and controller/device
  errors are not misses. Every result retains dense engine/control identity,
  terminal status, work statistics, and timing fields.

## Phase 11 standalone active-frontier push contract

- Frontier push is a separate engine, not a mode of dense push. Its canonical
  initial frontier contains every sorted, deduplicated source exactly once.
  Source labels are exact positive zero and all other labels are positive
  infinity. An empty initial frontier is invalid input, and targets never stop
  relaxation early. All canonical sources and targets remain one
  multi-source/multi-target query; they are not executed as separate searches.
- Retained scratch consists of one unsigned IEEE-754 distance word per vertex,
  two bounded 32-bit vertex queues, and one 64-bit enqueue-generation word per
  vertex. A zero requested capacity means `vertex_count`; any explicit
  capacity must be in `[1, vertex_count]`. The layout is checked for overflow,
  size, separation, and 64-bit generation alignment. Every generation word is
  reset to zero when a query is initialized.
- One executing thread owns each current-frontier entry and traverses that
  vertex's admitted outgoing CSR runs. A zero run mask skips the full run.
  Every admitted edge atomically loads the source label and applies the same
  strict unsigned-float atomic minimum as dense push. No CSC, prefix-sum edge
  balancing, virtual-warp/high-degree scheduler, wave aggregation, or adaptive
  scheduler is part of Phase 11.
- A successful strict decrease activates its destination for the next round.
  An atomic exchange of that destination's generation admits at most one queue
  claim for the round; later improvements still update its distance but count
  as duplicate suppressions instead of appending duplicate queue entries. The
  queue size is reserved atomically. A claim at or beyond capacity sets the
  explicit queue-overflow error without writing out of bounds.
- One controller transition owns each completed round. It consumes the
  materialized next-queue size, swaps read/write queue slots, clears the
  recycled size, and selects explicit overflow/device error, empty-frontier
  convergence, exact maximum-round exhaustion, or continued execution in that
  order. Empty-frontier convergence takes precedence when the final allowed
  round also exhausts the worklist. Distance slots and final status remain
  zero because the engine has one in-place label array.
- Ordinary execution uses one initializer, then ordered
  `(frontier_round_kernel, frontier_advance_kernel)` pairs, and one finalizer.
  Per-round control polls after one pair. Chunked control submits exactly `K`
  pairs for `K = 2,4,8,16,32` before polling; pairs already queued after
  device `done` are no-ops. Physical HIP controller copies are the initial H2D,
  `N` poll D2Hs, and terminal D2H (`N + 2`); HIP host synchronizations are
  `N + 1` because terminal D2H completion also retires the same-stream lease.
  Portable accounting intentionally counts semantic host
  observations instead of physical transfers.
- Persistent execution is exactly one cooperative query kernel containing
  initialization, run-mask construction, all queue rounds and swaps, terminal
  status, and uniform grid synchronization. It performs no host queue-size or
  convergence polling. Legal grids come from occupancy of the real persistent
  frontier kernel and selected block/shared-memory configuration; the Phase 6
  probe ceiling is not a frontier-kernel limit.
- Engine construction checks both resident shape and the deterministic
  two-word fingerprint of the exact host graph/run image. This rejects an
  accidentally paired, same-shaped resident upload; it is an accidental
  identity guard, not a cryptographic integrity or authenticity mechanism.
- Light instrumentation records admitted edges, strict decreases, active
  frontier vertices, executed/empty/small frontier rounds, and maximum queue
  size. The portable output also retains the exact current-frontier size for
  every executed round; the fixed HIP statistics ABI retains aggregate active
  vertices plus the executed-round count rather than a per-round device trace.
  Debug additionally records run-mask operations, atomic attempts and
  successes, queue claims, duplicate suppressions, and overflow events. None
  suppresses algorithm counters, while required control accounting remains in
  the result. Instrumentation must not alter final labels.
- Bounding-box miss is published only after empty-frontier convergence with an
  unreachable requested target. Maximum-round exhaustion, queue overflow,
  invalid controller state, and device failure are not misses. The current
  fixed host/device ABI is eight-byte aligned with `DeviceWorkStatistics` at
  168 bytes and `GpuSsspResult` at 224 bytes.
- All HIP compilation, device correctness, real occupancy, trace, profiler,
  timing, bounded-real, and full-graph evidence is deferred to the combined GPU
  campaign after all implementation phases.

## Phase 12 controlled single-query shootout contract

- The shootout compares exactly `JacobiPull`, `DenseChaoticPush`, and
  `FrontierPush`. It retains persistent cooperative control, per-round host
  polling, and chunked host polling at `K = 2,4,8,16,32`. It adds no fourth
  engine, hybrid, adaptive switching, batching, bounding-region expansion, or
  Phase 13 work.
- Canonical tuning identity contains one engine, one control mode, block size,
  chunk size when applicable, occupancy-derived or fixed-residency grid policy,
  fixed blocks per WGP when applicable, and maximum rounds. The required block
  sizes are 128, 256, and 512. Persistent catalogs contain the
  occupancy-derived choice plus every requested fixed blocks-per-WGP choice.
  Actual-kernel runtime limits decide legality; missing limits, illegal block
  sizes, and over-resident cooperative grids are recorded explicitly rather
  than silently omitted.
- A workload fingerprint binds two graph words, two query-corpus words, the
  corpus query count, and the shootout schema version. Every manifest and raw
  sample must carry that identity. Samples from another graph, query corpus,
  schema, run kind, or configuration catalog cannot be merged.
- Representative selection requires at least 1,000 eligible
  `logicnets_jscl` queries. It stratifies deterministically by selected-region
  vertices, selected-region edges, fanout, source count, and expected rounds.
  Expected rounds means the completed-round count from a deterministic Jacobi
  pilot and is selection metadata, never timing evidence. Selection records the
  seed, feature values, quantile bins, stable query IDs, and exact manifest.
  Named sparse-wavefront and dense-frontier adversarial cases are a separate
  required benchmark class and must not be substituted for the representative
  real-query minimum.
- Every configuration receives the identical graph, weights, selected region,
  canonical sources, and canonical targets for a query. A seed-controlled
  schedule interleaves query and configuration order for every retained
  repetition and assigns a unique global execution ordinal. Correctness groups
  configurations by query, with truthful reassigned ordinals, so its bounded
  Dijkstra oracle is built exactly once per query. Warmup precedes retained
  timing samples.
- Evidence stages are disjoint. Correctness samples use bounded Dijkstra,
  download final labels outside any timing interval, and must pass for every
  query/configuration pair before timing, counter, trace, or PMC evidence can be
  accepted. Final timing uses instrumentation `None`, a status-only result
  path, and no graph-sized label download. Algorithm work uses separate Debug
  runs. Trace and compatible PMC runs are separate from both.
- Timing records distinguish preparation GPU time, the SSSP device-timeline
  span, result-transfer GPU time, and end-to-end wall time. In host-poll modes
  the device-timeline span includes stream-idle gaps while the host observes
  and re-enqueues work; it is not summed kernel-active duration. Kernel-active
  durations require trace/profiler evidence.
- Reports retain P50/P95/P99 single-query wall and device-timeline time, total
  throughput with its wall boundary, rounds, examined edges, useful-decrease
  ratio, active/frontier work, atomic work, dispatches, host synchronizations,
  controller copies, real occupancy, L2 behavior, memory-unit activity,
  instruction mix, and P99 long-tail query identities and features. Missing
  profiler evidence is `unavailable`; a metric that has no meaning for a
  configuration is `not_applicable`; neither is encoded as a measured zero.
- Correctness and evidence completeness precede every conclusion. Each answer
  must name its workload and counter evidence. Until the deferred GPU campaign
  is collected, all seven requested performance questions and every default
  recommendation remain explicitly `pending measurement`; every engine,
  control, K, grid, and block-size toggle stays configurable.
- Under the maintainer execution-policy override, Phase 12 acceptance may use
  the full bounded CPU suite and source-level evidence. That acceptance is not
  representative-workload execution or GPU-performance evidence. Exact future
  evidence requirements are in `docs/PHASE12_GPU_VALIDATION.md`.

## Phase 13 deterministic overlapping-batch and workspace contract

- `SelectedRegionIndex` deep-validates the immutable graph/tile-run pairing
  once and coalesces exact directed tile-pair edge counts. Subsequent selected
  and union vertex/edge estimates traverse selected tile spans and indexed tile
  pairs; they do not rescan every graph vertex or edge for every query.
- Planner features retain query ID, expansion generation, source/target counts
  and owner-tile sets, actual selected tiles, and exact selected vertex/edge
  estimates. Feature construction canonicalizes by the full 32-bit `QueryId`
  domain, so input permutation does not change the semantic plan or stable TSV.
- The standard family is ordered width 32, 16, then 8. Width 64 remains a
  separate future two-wave experiment. Every query appears in exactly one
  valid low-prefix lane; a partial final batch is padded with invalid identities
  and zero generation/count payloads, and singleton leftovers are retained.
- The default overlap gate is tile-set Jaccard at least `1/8`; the default
  projected union-inflation gate is at most `2/1`, where inflation is
  `valid_lanes * union_tile_count / sum(selected_tile_count_by_lane)`. An anchor
  is chosen by descending selected edges, selected vertices, and tile count,
  followed by ascending generation and query ID. Candidate preference is
  descending Jaccard against the accumulated union, descending matching
  source-plus-target tile overlap, ascending projected inflation, union edges,
  union vertices, new tiles, generation/source/target-count differences, and
  query ID. Exact integer fraction comparison makes ties platform-independent.
- A batch description contains QueryIds and generations by lane, independently
  offset source and target arrays, selected estimates, sorted union tiles,
  dense per-tile lane masks, selected vertex ranges, zero initial reached/miss
  masks, an optional dense per-tile compact-index bias map, and one of two
  exact run representations. Compact run descriptors include per-union-vertex
  CSR/CSC offsets. CSR run masks are
  `source_owner_mask & destination_tile_mask`; CSC masks are
  `destination_owner_mask & source_tile_mask`. Zero-mask runs are omitted from
  compact descriptors or left zero in retained storage and never require
  repeated endpoint admission per edge.
- Deep validators prove canonical padding, identity, once-only assignment,
  union/mask/estimate reconstruction, the exact greedy result, terminal
  payloads, vertex ranges, run ordering, per-lane admitted-edge totals, and
  endpoint-by-endpoint admission equivalence on bounded fixtures. All size,
  byte, offset, and accumulation arithmetic is checked before reservation. A
  serialized workspace decision additionally binds immutable graph vertex/tile
  counts and the complete 2x2 strategy matrix so coordinated row-only
  mutations that disagree with those recorded counts fail.
- The workspace model compares the cross product of full graph-sized versus
  compact union-tile vertex storage and retained per-run masks versus compact
  nonzero run descriptors. Retained reservations grow geometrically and reuse
  capacity; retained-mask preparation clears only the previous touched ledger.
  Cold initialization and warmed clear/write traffic are distinct, and each
  reusable batch image is bound to its first run representation.
  `docs/PHASE13_WORKSPACE_DECISION.md` records the assumptions and evidence.
- The bounded-synthetic provisional choice is full graph-sized vertex-major
  lane storage plus retained per-run masks because it has no global/local label
  mapping and has the simpler reusable preparation path on the acceptance
  fixture. This is not a production default: the real `.bfqueries` artifact,
  real union/run counts, runtime free memory, HIP compilation, batched kernels,
  and device preparation/timing measurements are absent.
- Phase 13 implements no HIP source, batched relaxation, per-lane convergence,
  expansion/replanning execution, congestion state, or GPU performance path.
  Phase 14 consumes its exact batch descriptions without changing the frozen
  Phase 13 planning result.

## Phase 14 overlapping batched Jacobi-pull contract

- Batched Jacobi supports widths 1, 8, 16, and 32. The standard overlap
  planner remains restricted to 32, 16, and 8; width one is an explicit
  singleton correctness/baseline path and does not change Phase 13 policy.
- Each distance slot is vertex-major with contiguous query lanes:
  `index = vertex * lane_width + lane`. Relaxation uses two distinct slots,
  ordinary float addition, strict `<`, and the same canonical incoming CSC
  edge order as standalone Jacobi. Width one must therefore agree bitwise with
  the standalone engine.
- Flattened source and target payloads remain partitioned by their per-lane
  offsets. Every source in a valid lane is seeded independently to exact
  positive zero in both slots; sources from different lanes are never merged.
  Padded lanes have empty terminal slices and perform no semantic work.
- A selected vertex range supplies the destination admission mask. A prepared
  CSC source-tile run is intersected once with the current execute mask; a
  zero result skips the complete run, while every edge record in a nonzero run
  is shared across its admitted query lanes. The portable path accepts both
  exact Phase 13 host run representations. The production HIP path keeps the
  dense CSC mask only on the device and wave-cooperatively materializes every
  selected destination-column run during Jacobi initialization as destination
  mask intersected with source-tile mask. It overwrites zero as well as
  nonzero selected entries, never reads stale nonselected entries, and performs
  no graph-wide mask clear or host-to-device run-mask transfer per batch.
- One complete round copies every selected destination for each executing lane
  into the write slot, including unchanged values. The shared controller swaps
  the two slots only after that complete copy. With per-lane convergence
  enabled, a no-change lane is then frozen and removed; its selected-region
  columns are already bitwise identical, so later batch-wide swaps are safe.
  With convergence disabled, valid lanes continue through the batch's final
  global no-change round. Both settings must produce identical final labels.
- Per-lane convergence defaults to enabled. A one-based first no-change round,
  executed rounds, and later tail rounds are retained independently per valid
  lane. Maximum-round exhaustion is not convergence and does not fabricate a
  first no-change round for an unfinished lane.
- Ordinary per-round control launches one relaxation/advance pair before each
  poll. Chunked control submits exactly `K` ordered pairs and permits queued
  post-`done` pairs to no-op without changing real-round parity. Persistent
  control performs initialization, all rounds, controller transitions, and
  final status in one cooperative device-controlled kernel with uniform grid
  barriers and no convergence polling.
- Reached/miss masks are classified only after normal convergence. A lane is
  reached only when every target in its own target slice is finite; otherwise
  it is a bounded-region miss. Maximum-round and controller/device-error exits
  publish neither mask. Targets never terminate relaxation early.
- Batch work statistics distinguish physical CSC edge-record loads from
  logical lane-edge relaxations; considered, visited, skipped, and active CSC
  runs; active lanes across visited runs; destination/lane writes and
  decreases; valid, padded, active, and tail lane-round capacity; exact
  per-lane early-convergence work avoided; union-tile inflation counts; and
  modeled selected two-slot device reset/source-seed bytes. Fresh portable
  vectors may initialize nonsemantic host-container cells as a convenience;
  those writes are not reported as device reset traffic. All host accumulations
  are checked before overflow. None of these portable counters is GPU timing or
  cache evidence.
- The HIP mapping assigns one wave32 to a destination and query lanes to wave
  lanes. Block size must be wave32-compatible, and kernel occupancy must be
  queried from the actual selected batch kernel. Compiler-generated uniform
  loads are the default. An explicit `__shfl` broadcast variant remains
  selectable but unselected until the two paths are measured on the same
  target, workload, control, and instrumentation level.
- A batch-specific HIP workspace owns checked retained buffers for terminal
  data and offsets, selected ranges and their packed offsets, tile masks, a
  device-only dense CSC run-mask image, controller/status/instrumentation
  records, convergence rounds, statistics,
  and two full graph-sized vertex-major distance slots. It retains capacity
  geometrically and binds preparation, execution, result copies, and
  retirement to one stream. Initialization touches selected range/lane cells;
  nonselected and padded cells are nonsemantic.
- Selected ranges are the authoritative device union description, so Jacobi
  preparation uploads no separate union-tile array. The initializer writes the
  complete controller before any use, and finalization writes the complete
  status before readback; preparation therefore performs neither an initial
  controller H2D nor a status clear. For `N` ordinary polls, full/evidence mode
  has `N + 1` controller D2Hs and `N + 1` host synchronizations. The final D2H
  completion is also the same-stream lease-retirement boundary.
- Bounded CPU tests are the local Phase 14 acceptance evidence. They must cover
  widths 1/8/16/32, independent and multi-source lanes, padding, immediate/
  one-round/several-round/unreachable cases, early freeze across later slot
  swaps, both convergence settings, all three controls, endpoint-exact masks,
  counters, maximum-round exits, width-one standalone agreement, and
  independent bounded Dijkstra comparisons.
- HIP compilation, device execution, real-corpus correctness, explicit-load
  comparison, register pressure, actual occupancy, L2/read traffic, latency,
  throughput, and batching benefit are deferred under the current execution
  policy. Missing hardware evidence is unavailable, not a measured zero. The
  exact future procedure is `docs/PHASE14_GPU_VALIDATION.md`.
- Phase 14 itself implements no dense batched push, expansion/replanning
  execution, path reconstruction integration, congestion update, or
  production tuning recommendation. Phase 15 now implements overlapping
  batched dense chaotic push as a separate engine.

## Phase 15 overlapping batched dense-chaotic-push contract

- Batched dense push supports widths 1, 8, 16, and 32. Width one is an
  explicit singleton correctness/baseline path; the standard overlap planner
  remains the Phase 13 width-32/16/8 family.
- Distance state is one graph-sized, vertex-major array of unsigned IEEE-754
  words: `index = vertex * lane_width + lane`. Each lane owns independent
  state, its own offset source/target slices, and exact positive-zero seeds.
  Cross-lane source merging is forbidden. Padded lanes have no terminals,
  selected-mask bits, status bits, atomic attempts, or semantic work.
- The supported atomic domain is the standalone dense domain: finite
  nonnegative floats, canonical positive zero, and positive infinity. Every
  admitted lane-edge atomically loads its own source word and applies a strict
  unsigned-word atomic minimum to its own destination word. Ordinary loads may
  not race atomic writes. Dense batching has no frontier, queue, predecessor,
  second distance slot, or target-based early termination.
- A selected source row supplies the current source-lane mask. Its prepared
  CSR destination-tile run mask is intersected once with the execute mask. A
  zero result skips the complete run; otherwise the run mask is reused for
  every edge record. The portable path accepts retained masks and compact
  nonzero descriptors. The initial HIP path consumes only the provisionally
  selected retained CSR mask image.
- One round is one complete admitted-edge scan. With per-lane convergence
  enabled, a lane freezes only after a complete scan produces no strict
  decrease for that lane. With convergence disabled, all valid lanes continue
  through the batch's final global no-change scan. Both settings must produce
  identical selected-region distance bits. A genuine first no-change scan,
  executed scans, and later tail scans are retained independently per lane;
  maximum-round exhaustion does not fabricate convergence.
- Ordinary per-round control launches one dense-scan/controller-advance pair
  before each poll. Chunked control submits exactly `K` ordered pairs for
  `K = 2,4,8,16,32`; queued pairs after device `done` are no-ops. Persistent
  control performs initialization, every scan, controller advancement, and
  terminal status inside one cooperative device-controlled kernel with uniform
  grid barriers and no convergence polling.
- Reached/miss masks are produced only after normal convergence by checking
  each lane's own target slice. Maximum-round and controller/device-error exits
  publish neither. Width one must agree bitwise with standalone dense push,
  and every valid lane must agree with an independent bounded Dijkstra query.
- Batch counters distinguish algorithmic CSR edge-record requests from logical
  lane-edge relaxations, atomic-compatible source loads and destination-min
  attempts, useful updates, considered/visited/skipped runs, active source/lane
  work, full scans, and early-convergence tail work. `csr_edge_loads` is a
  logical request count; it is not a measured physical cache load or L2
  transaction. Exact integer capacity terms separately expose valid-lane,
  configured-width, wave32, unused-wave, padded-lane, and edge-wave lane
  capacity. CPU counts are not GPU occupancy, cache, write-stall, or timing
  evidence.
- `high_contention_destinations` remains a standalone dense-engine diagnostic;
  Phase 15 batched dense does not compute it. A zero in the shared result
  structure means unavailable for this engine, not a measured absence of
  contention.
- Reset accounting models one selected 32-bit word per admitted vertex/lane;
  source seeds are the zero-valued subset of those initialization writes.
  Fresh portable full-vector construction may initialize nonsemantic host
  cells, but those convenience writes are not presented as device traffic.
- `hip::BatchedDenseChaoticPushEngine` accepts a validated
  `BatchDeviceDescription`; `ReusableBatchedDenseWorkspace` owns checked
  retained terminal, selected-range, retained-CSR-mask, controller/status,
  optional statistics, convergence-round, and one-slot distance buffers. A
  lease binds preparation, execution, result transfer, and retirement to one
  stream.
- The HIP mapping assigns one wave32 to a selected source row and query lanes
  to wave lanes; that wave services the row's outgoing CSR edge requests.
  Block size must be wave32-compatible, and launch legality must be queried
  from the selected batched-dense kernel. The default
  `BatchedDenseLoadStrategy::compiler_uniform` leaves immutable edge fields as
  compiler-visible uniform loads. `explicit_wave_broadcast` is an unselected
  experiment until a correctness-matched target-device comparison exists.
- Bounded CPU tests are the local Phase 15 acceptance boundary. They cover
  widths 1/8/16/32, retained/descriptor parity, independent and multi-source
  lanes, padding, no-first-decrease/short/several-scan/unreachable cases, both
  convergence settings, persistent/per-round/chunked controls, exact counters,
  instrumentation separation, maximum-round exits, width-one standalone
  parity, cross-lane isolation, and bounded Dijkstra comparisons. The final
  Release suite passed `16/16` tests in `0.17` seconds; the focused Phase 15
  CTest passed `1/1` in `0.01` seconds; and the ASan+UBSan suite passed
  `16/16` in `1.82` seconds, including Phase 15 `1/1` in `0.14` seconds.
- HIP compilation, device execution, device correctness, a real-corpus run,
  compiler-uniform/broadcast comparison, register pressure, actual occupancy,
  L2 reads/writes, write stalls, latency, throughput, and batching benefit are
  deferred to the combined campaign after all implementation phases. Missing
  evidence is unavailable, not a measured zero. The exact future procedure is
  `docs/PHASE15_GPU_VALIDATION.md`.
- Phase 15 itself implements no batched frontier push, expansion/replanning
  execution, path reconstruction integration, congestion update, adaptive
  engine, or production tuning recommendation. Phase 16 now adds overlapping
  batched frontier push separately.

## Phase 16 overlapping batched active-frontier-push contract

- Batched frontier push supports widths 1, 8, 16, and 32. Width one is an
  explicit singleton correctness/baseline path; the standard overlap planner
  remains the Phase 13 width-32/16/8 family.
- Distance state is one graph-sized vertex-major array of independent unsigned
  IEEE-754 words at `vertex * lane_width + lane`. Every valid lane retains its
  own source/target slices, positive-zero seeds, selected tiles, and result
  bits. Shared source vertices may share one initial queue entry, but source
  sets and distance words are never merged. The production input guard rejects
  noncanonical terminal slices, duplicate valid-lane query IDs, and semantic
  metadata or terminal payload in padded lanes. Padding performs no semantic
  work.
- Frontier state is two bounded 32-bit vertex queues and two graph-sized
  `LaneMask` arrays. A queue entry represents one vertex active for at least
  one query lane. The zero-to-nonzero transition of the destination's next
  mask owns its one queue claim; later successful updates may add lane bits or
  improve labels without appending a duplicate vertex. A zero requested queue
  capacity means `vertex_count`; an explicit smaller capacity is only an
  overflow-validation seam. Overflow is terminal, explicit, and never writes
  beyond the queue.
- One simple grid-stride thread owns one current worklist entry, matching the
  Phase 11 scheduling policy. It reads and clears that vertex's current mask,
  intersects each outgoing CSR destination-tile run with the current and
  prepared admission masks once, skips a zero result, and reuses a nonzero mask
  for every edge in the run. Every admitted lane uses an atomic-compatible
  source load and strict destination minimum. The successful lane bits for an
  edge are atomically ORed into the destination's next mask before its optional
  queue claim.
- Round-boundary invariants require unique in-range queue vertices, nonzero
  per-entry masks, exact agreement between the queue and nonzero mask image,
  and equality between the OR of per-vertex masks and the controller frontier
  lane mask. Consumed current masks are zero before that slot is recycled.
  Queue entries count vertices, never vertex/lane pairs. The portable engine
  validates the full invariant set at every bounded round boundary; its result
  exposes the final queue/mask images plus per-round current-size and lane-union
  traces. The current HIP result and deferred HIP CTest do not expose a complete
  per-round queue-to-mask boundary ledger; device proof of the full invariant
  set requires a future or manual Debug diagnostic extension in the combined
  campaign.
- A batched-frontier-specific controller transition increments only a completed
  round, swaps the queue slots, clears the recycled size and next-round state,
  and gives explicit errors precedence before empty-frontier convergence and
  maximum-round exhaustion. Empty-frontier convergence wins when the final
  allowed round also exhausts the queue. The standalone Phase 11 transition is
  not reused because its one-lane active-mask rules do not represent a batch.
- With per-lane convergence enabled, a lane absent from the complete next
  frontier moves from active to converged and its distance state remains
  frozen. With convergence disabled, that absent lane receives no synthetic
  queue work: the execute mask still follows actual next-frontier presence,
  while controller-level convergence is deferred until the complete batch
  frontier is empty. Both settings therefore produce identical final bits.
  The first one-based empty-next-frontier round, executed rounds, and later
  batch tail rounds remain independently visible. Under this required queue
  policy, the semantic work avoided solely by changing the convergence flag
  may correctly measure zero. Initialization resets every configured lane's
  convergence record with a grid-stride loop. A record is published only for a
  clean accepted continue, convergence, or maximum-round transition; error,
  invalid-controller, and terminal no-op transitions publish no convergence
  proof.
- Ordinary control executes ordered frontier-round/controller-advance pairs.
  Per-round polling observes after one pair. Chunked control queues exactly
  `K = 2,4,8,16,32` pairs before observing, and pairs after device `done` are
  no-ops. Persistent control performs initialization, all queue rounds and
  swaps, and terminal status in one cooperative kernel with uniform grid
  barriers and no host frontier/convergence polling. Ordinary launches expose
  the occupancy- and queue-work-capped block count as
  `ordinary_grid_blocks`; this field is launch metadata, not measured
  occupancy.
- Reached/miss masks are classified only after normal whole-batch convergence
  by checking each valid lane's own target slice. A finite value for every
  target means reached; otherwise the lane is a bounded-region miss.
  Maximum-round, overflow, invalid-controller, and device-failure exits publish
  neither result mask. Targets never terminate relaxation early.
- Batch work records distinguish physical worklist vertices from active
  vertex/lane pairs, shared CSR edge-record requests from logical lane-edge
  relaxations, multi-lane sharing, run visits/skips, active lanes across runs,
  distance atomic attempts/successes, next-mask atomics, unique lane
  activations, queue claims, queue entries saved by lane merging, duplicate
  suppressions, high-water mark, overflow, utilization capacity, lane tails,
  and selected distance/mask reset traffic. These are algorithmic counts, not
  measured physical cache traffic, occupancy, or timing. `None` publishes no
  device algorithm counters. `Light` publishes only aggregate work, sharing,
  and queue-high-water evidence. `Debug` additionally publishes mask atomics,
  unique activations, queue claims, lane-merging savings, same-lane and total
  duplicate suppressions, and overflow events.
- `hip::BatchedFrontierPushEngine` is batch-specific rather than a
  `GpuSsspEngine::run` adapter. Its dedicated
  `ReusableBatchedFrontierWorkspace` retains terminal offsets, selected ranges,
  exact retained CSR masks, controller/status/statistics records,
  per-lane convergence rounds, one distance array, two queues, and two activity
  masks under one stream lease. Preparation validates canonical source/target
  slices, unique valid-lane query IDs, and empty canonical padding before any
  launch. The first HIP path deliberately consumes the provisional full-
  vertex/retained-CSR representation; portable compact-descriptor equivalence
  does not implement compact device storage.
- The device relaxation keeps the Phase 11 one-thread-per-worklist-entry
  scheduler. Prefix-sum edge balancing, virtual-warp/high-degree scheduling,
  wave-aggregated queue insertion, and adaptive frontier scheduling remain
  documented future work and are not implemented in Phase 16. There is no
  fallback to dense chaotic push and no adaptive engine combination.
- Bounded CPU tests are the local Phase 16 acceptance boundary. They cover the
  width/control/convergence matrix, exact retained/descriptor parity across
  the whole configured `V * W` image, independent and multi-source lanes,
  shared-source and shared-destination queue merging, padding, mixed frontier
  durations, unreachable targets, high fan-in and repeated improvements,
  zero-weight cycles, run admission/reuse, queue and mask invariants,
  initialization- and round-time overflow, instrumentation separation,
  clean maximum-round and error convergence-record separation, width-one
  standalone parity, cross-lane isolation, and independent bounded Dijkstra
  comparisons. Every padded lane and every
  nonselected valid-lane cell in the public full output is positive infinity;
  this normalization does not enlarge modeled selected device-reset traffic.
  Final post-hardening HIP-off evidence is Release `17/17` in `1.56` seconds,
  including Phase 16 `1/1` in `0.02` seconds, and ASan+UBSan `17/17` in `3.03`
  seconds, including Phase 16 `1/1` in `0.16` seconds. The deferred HIP test
  passed strict-warnings host syntax and fake-HIP `__HIPCC__` syntax, including
  its device probe. Phase 16 public headers passed strict host syntax, and both
  production HIP translation units passed strict fake-HIP syntax. These are
  source checks, not HIP compilation or device execution; no browser, cloud
  service, or large/full-graph test was used.
- HIP compilation, device execution, device correctness, real-corpus evidence,
  actual occupancy, physical memory traffic, latency, throughput, batching
  benefit, and comparisons with batched Jacobi or dense push are deferred to
  the combined campaign. Missing evidence is unavailable, not measured zero.
  The exact later procedure is `docs/PHASE16_GPU_VALIDATION.md`.
- Phase 16 itself implements no failed-lane expansion, query
  replanning/restart, full-region fallback, or all-query execution. Phase 17
  now adds those operations above all three completed batched engines.

## Phase 17 reachability expansion and all-query contract

- `run_batched_expansion` is the engine-independent all-query controller. It
  validates the spatial graph, directory, tile runs, queries, engine options,
  and planner policy; sorts state by `QueryId`; plans every active query; and
  invokes one batch callback at a time. A clean converged compact status must
  partition every valid lane into exactly reached or bounding-region miss.
  Reached lanes become terminal immediately. All misses from the completed
  planning pass are collected before any expansion, then deterministically
  expanded, replanned, and restarted together.
- Four explicit schedules exist: one geometric ring per retry; a configured
  fixed additive ring of at least two; absolute x/y margins `1,2,4,...`; and a
  configured number of additive one-ring steps followed by margin doubling.
  A schedule is mandatory. There is no compiled production default.
  `select_expansion_schedule_from_evidence` accepts only one comparable record
  for each schedule, with the same nonzero workload/configuration fingerprint,
  and rejects zero-query, duplicate, incomplete, malformed, or mismatched
  evidence. The fingerprint includes a required caller-owned identity for
  execution settings outside the shared engine/planner options. The selector
  returns no recommendation when the best evidence score is tied.
- Geometric expansion is anchored to the original selected located bounds and
  preserves already selected tiles. The intended margin advances according to
  the schedule, then admits every represented located tile inside that box.
  Expansion must strictly add a tile. A stalled region either enters the one
  configured full-region fallback or terminates explicitly. Spill has no
  synthetic coordinate: it is preserved when selected and otherwise enters
  only through actual tile-directory adjacency. Long cross-tile edges remain
  governed by endpoint admission and do not create synthetic intermediate
  rings.
- Every retry preserves QueryId, source/target terminal arrays, canonical
  solver sets, and terminal maps. It increments `expansion_generation`,
  discards the previous label image, rebuilds overlap features and run masks,
  and invokes the selected engine's normal selected-only reset and original-
  source seed path. Failed labels, queues, masks, changed flags, controllers,
  and convergence evidence are never reused. A valid `UINT32_MAX` QueryId is
  distinct from padding through the valid-lane mask. A query whose generation,
  retry/expansion count, or margin cannot advance terminates as
  `identity_or_count_overflow` without aborting other queries. Aggregate
  telemetry, histogram, or campaign-count overflow instead throws and aborts
  the campaign fail-closed. Neither boundary may wrap.
- `maximum_expansions` limits scheduled geometric expansions. A full-region
  fallback, when configured, is one separate final restart and occurs at most
  once. A miss in that full region terminates as
  `unreachable_in_full_region`. The alternative explicit policy distinguishes
  `expansion_limit` from `region_stalled`. No terminal failure is reported as
  success.
- Only clean whole-batch convergence produces reachability decisions.
  Maximum-round exhaustion, queue overflow, invalid-controller state, or
  device failure terminalizes affected lanes as `engine_failure` without
  expansion. A nonclean status carrying reached/miss bits is a malformed runner
  contract and is rejected. No target-reached lane is expanded, and no exit
  lower-bound certificate is introduced: a reached target remains accepted on
  its current induced region.
- Final outcomes are in canonical QueryId order and carry the terminal query,
  disposition, attempts, scheduled and total expansion counts, fallback flag,
  selected-region sizes, terminal status/error, and an optional graph-sized
  lane distance projection. The deterministic trace records planning pass,
  retry flag, batch/lane identities and generations, masks, union/selected
  estimates, work evidence, and terminal status. Host-wall timing is excluded
  from trace identity.
- Metrics report initial and final success, every failure disposition, the
  expansion histogram, failed-origin and retry lane utilization, scheduled
  expansions and fallbacks, repeated selected-edge estimate, attempted/final
  region sizes, planning/replanning/execution/total host time, integer all-
  query throughput, aggregate device work, and actual measured retry and
  failed-batch shared/logical edge work. `unavailable` work evidence must carry
  no numeric work and is never interpreted as measured zero. Aggregate metric
  accumulation is checked; overflow is a campaign error, not a per-query
  disposition.
- `run_host_batched_expansion` is the bounded portable adapter. On every
  callback it rebuilds one `BatchDeviceDescription` and invokes exactly one of
  the Phase 14 Jacobi, Phase 15 dense, or Phase 16 frontier host engines from a
  fresh engine reset. Retained run masks and compact descriptors are supported
  through the existing batch-preparation representations. The adapter folds
  that representation, dense schedule, and frontier queue capacity into the
  caller's nonzero execution-configuration fingerprint.
- HIP-gated `hip::BatchedExpansionExecutor` binds one underlying engine,
  resident graph, reusable engine workspace, stream, and immutable run
  options. Compact production transfer requires `InstrumentationLevel::none`
  and copies exactly one 48-byte `DeviceRunStatus` after the existing engine
  finalizer; it downloads no counter block, lane trace, controller, or `V * W`
  labels. Evidence transfer retains the prior status-only paths and labels
  shared/logical work explicitly measured or unavailable. The adapter adds no
  relaxation kernel or expansion workspace. Transfer/load strategy is part of
  the required execution-configuration identity used for schedule evidence.
- `bfnew_gpu_batched_expansion` is the HIP+FPGAIF-gated deferred campaign
  driver. It requires explicit versioned graph/query artifacts, a fresh output
  path, engine, and schedule; exposes every schedule parameter, terminal
  policy, control/grid/convergence option, and compact/evidence transfer mode;
  uploads one resident graph; executes every artifact query; validates the
  canonical terminal ledger; and publishes a fail-closed
  `bfnew.batched-expansion.v1` TSV through a temporary file and atomic rename.
  The report carries graph/query/layout, execution-configuration, and schedule-
  comparison fingerprints, complete configuration, aggregate/expansion/work
  metrics, canonical query outcomes, selected tiles, and retry batch traces.
  It refuses to overwrite evidence and returns failure for engine or per-query
  identity/count failures; aggregate telemetry overflow aborts before a report
  can be accepted. The driver synchronizes resident upload before the
  controller timer but begins with a cold reusable workspace, so the timer
  includes first workspace growth and is diagnostic cold-controller evidence,
  not warm final throughput. It has no label-download/Dijkstra mode and is not
  by itself the final timing or profiler harness.
- Bounded HIP-off tests cover all four schedules across all three real portable
  engines; several misses and exactly one miss; dissimilar replans; long-edge,
  spill, multi-source, multi-retry, fallback, explicit failure, stalled, and
  globally unreachable cases; deterministic order; nonzero and maximum
  identities/generations; overflow and malformed runner exits; retained/
  descriptor parity; exact metrics; execution-configuration binding; invalid-
  campaign and evidence-score-tie rejection; and bitwise induced-region
  Dijkstra agreement. The exact local target is
  `bfnew.batched_expansion`. Final HIP-off evidence is Release `18/18` in
  `1.70` seconds, including Phase 17 `1/1` in `0.01` seconds, and ASan+UBSan
  `18/18` in `3.32` seconds, including Phase 17 `1/1` in `0.19` seconds.
  Strict host and fake-HIP syntax checks are source evidence only and used no
  HIP compiler or GPU.
- HIP compilation of the executor or campaign driver, device execution,
  compact-transfer validation, real-corpus
  expansion distributions, actual replanning cost, target occupancy/traffic,
  latency, throughput, and the schedule choice remain deferred to the combined
  campaign in `docs/PHASE17_GPU_VALIDATION.md`. The required
  `logicnets_jscl.padding1.v1.bfqueries` artifact is absent, so no schedule is
  selected from the bounded synthetic fixture.
- Phase 17 stops before compact target/path results and reconstruction
  integration. Phase 18 now implements that post-relaxation boundary without
  adding congestion updates, an exit certificate, or adaptive engine
  selection.

## Phase 18 compact target/path and no-congestion contract

- `CompactTargetSummary` is a fixed-width, trivially copyable 28-byte record.
  `path_length` is an edge count. `has_selected_source` is exactly zero or one;
  no strong `VertexId` value is reserved as a sentinel. A complete target has
  a finite distance, reached status, valid canonical query source, and complete
  reconstruction status. An unreachable or terminal-failure target has no
  invented source or path.
- `CompactTargetPath` stores `path_length + 1` global reordered vertex IDs and
  their actual final lane-distance labels plus `path_length` stable logical
  edge IDs. Those arrays are path-sized output, not predecessor state and not
  a graph-sized label transfer. `CompactPathPayload` binds the target arrays to
  QueryId, expansion generation, canonical target order, and the unchanged
  original-terminal map before a runner may reuse its workspace.
- `reconstruct_compact_query_paths` is the bounded portable semantic
  implementation. It reads final distances and incoming CSC after relaxation;
  it never writes or retains a relaxation predecessor. An incoming candidate
  is tight only when ordinary float `predecessor_distance + weight` equals the
  current label exactly. Tight candidates are tried in stable `EdgeId` order.
  Path-local cycle detection and real backtracking escape zero-weight cycles
  and lower-ID tight dead ends. A target that is itself a canonical source has
  one vertex, one zero label, and zero edges.
- Reconstruction admits only vertices whose owner tiles occur in the query's
  final selected region. Every returned edge therefore has both endpoints in
  that induced subgraph. Reached bounded paths remain accepted even when
  unbounded Dijkstra finds a cheaper route. Path-quality inflation is report
  data, never an expansion trigger or correctness failure.
- `validate_compact_query_result` independently checks identity, generation,
  terminal map, target order, statuses, selected source, shapes, simple source-
  to-target termination, selected-region membership, stable-edge continuity,
  exact compact-label tightness, and ordinary-float accumulated cost.
  `validate_compact_query_result_against_distances` additionally binds a
  bounded diagnostic full image; production execution does not obtain a full
  image merely to call the ordinary validator.
- `measure_compact_transfer` reports checked summary, vertex, path-label,
  stable-edge-ID, and total result bytes. Query metadata and terminal maps are
  already host-resident and are excluded. Device vertices and stable edge IDs
  are checked 32-bit words and each occupy four serialized payload bytes; host
  `EdgeId` widening does not change that model. Physical D2H claims additionally
  require the HIP compact payload/status/error subtotal, controller-poll count
  and bytes, overall D2H total, and trace. Persistent control has zero
  controller-poll traffic; ordinary per-round and chunked controls report it
  separately rather than hiding it in the compact subtotal, with poll bytes
  exactly `poll_count * sizeof(DeviceController)` (96 bytes per poll).
  Aggregate byte/count overflow is a campaign failure and may not wrap or be
  presented as measured zero.
- Phase 17 execution has an explicit `enable_compact_paths` word. It must be
  zero or one. When enabled, every valid lane in a clean batch publishes one
  generation-matched payload before workspace reuse. Reached lanes contain
  only complete targets. A miss may contain complete and unreachable target
  summaries but no false reached classification. Before retry, every clean
  miss must contain at least one unreachable target and every other target
  must be complete or unreachable; an all-complete miss or any reconstruction
  failure becomes an explicit engine failure instead of being expanded.
  Installing a valid retry discards that generation's labels and payload. A
  terminal full-region miss retains its classified payload. Missing, padded,
  duplicate, or wrong-generation payloads are malformed, and a reached status
  paired with an incomplete payload becomes an explicit engine failure.
- `extract_host_compact_paths` is the HIP-off adapter. It consumes an already
  captured payload when present or independently reconstructs from the
  portable terminal projection, creates explicit results for terminal
  failures, and releases graph-sized correctness images after compaction. The
  returned result type contains no lane-distance matrix.
- The production HIP path reconstructs after the selected Phase 14--16 engine
  finalizer and before workspace retirement. It transfers fixed target
  summaries followed only by exact compact path arenas. It must not copy a
  `vertex_count * lane_width` distance image to the CPU. The same final label
  slot used for terminal classification is the reconstruction input; failed
  generations and inactive/padded lanes produce no complete payload.
- Every sampled compact path must validate source termination, target
  termination, stable-edge continuity, exact tightness relative to its actual
  final GPU labels, reported cost, and final selected-region membership.
  Diagnostic full-image replay remains outside timed production execution.
- The complete no-congestion campaign loads the versioned graph/query
  artifacts, uploads the immutable graph once, plans every query, runs exactly
  one named SSSP engine/control configuration, performs Phase 17 expansion and
  restart, reconstructs compact terminal paths, and returns a canonical result
  or explicit failure for every query. It does not silently select an engine,
  control mode, expansion schedule, or adaptive combination.
- Timing reports cold artifact load, graph upload, all initial/retry planning,
  SSSP, geometric region growth, controller/orchestration, reconstruction,
  compact result transfer, total warm all-query time, a separate first
  capacity-growing cold-execution interval, and total cold-pipeline time as
  separate stages. Host wall and applicable GPU-event clocks remain distinct.
  Warm performance uses a resident graph and pre-grown reusable capacity; a
  cold first allocation is not relabeled as warm throughput.
- The Phase 18 path campaign accepts lane widths 1, 8, 16, and 32; width 1 is
  the scalar baseline, not padding or a different correctness mode. Every
  cold, additional-warmup, and measured-repetition execution must carry
  measured SSSP, reconstruction, and result-transfer GPU-event evidence. A
  missing device observation fails the campaign rather than becoming zero or
  an omitted sample.
- `NoCongestionStageLedger` records timing evidence separately from numeric
  values. An unavailable host/device observation must carry zero, every
  measured device duration must be finite and nonnegative. When all named host
  stages are measured, `warm_all_query` is their checked exact nonoverlapping
  sum after reusable capacity is already grown. An asynchronous HIP execution
  may instead leave host-unpartitionable named stages unavailable and supply a
  separately measured enclosing `warm_all_query`; unavailable is not measured
  zero and those costs are not folded into another named stage.
  `cold_execution` is an independent enclosing observation of the first all-
  query execution after upload and may include workspace construction and
  capacity growth. `cold_pipeline` is the checked exact sum of artifact load,
  graph upload, and that distinct `cold_execution`; no equality between cold
  and warm execution is asserted. Aggregate device time is never synthesized
  by adding incomparable events.
- `NoCongestionResultAccounting` partitions canonical targets by complete,
  unreachable, terminal-failure, or reconstruction-failure status. Its
  portable fields model final serialization using one 48-byte status record
  per declared batch, 28-byte summaries, checked-u32 vertices/edges, and float
  labels; they exclude discarded retry payloads and are not physical D2H
  evidence. When HIP metrics are supplied, separate evidence fields preserve
  the cumulative compact payload, status, error, compact-subtotal,
  controller-poll count/bytes, and overall device-transfer bytes, including
  retries. The overall value is the checked compact subtotal plus controller-
  poll bytes. `run_host_no_congestion_pipeline` exercises the same generation-
  bound handoff and returns no graph-sized lane matrix.
- `sample_compact_path_quality` selects queries independently of input order by
  SplitMix64 over QueryId and a recorded seed. It compares complete bounded
  targets, including complete targets in a mixed terminal full-region result,
  with unbounded multi-source Dijkstra. It reports absolute inflation and
  nearest-rank P50/P95/P99/maximum cost/path-length ratios. A zero/zero baseline
  has unit ratio; a positive bounded cost over a zero unbounded cost has
  positive-infinite ratio and finite absolute inflation. This sample is quality
  evidence only and may not alter reachability, expansion, or correctness.
- Representative subsets cover all three engines with persistent,
  chunked-host-poll, and per-round-host-poll controls. The recommended
  configuration may run over all available queries only after complete
  comparable evidence selects it. The absent
  `logicnets_jscl.padding1.v1.bfqueries` artifact and absent device execution
  leave this real campaign and its recommendation deferred.
- Bounded HIP-off coverage uses fractional, unequal, and zero weights;
  stable-ID equal paths and equal-cost parallel edges; zero-cycle backtracking;
  target-is-source and
  multi-source selection; duplicate terminal maps; a bounded route with a
  cheaper excluded global path; long and spill paths; retry generations;
  clean unreachable and engine-failure records; transfer-byte accounting;
  timing-evidence and checked-sum failures; deterministic QueryId-hash quality
  sampling; validation corruption; padded lanes; and deterministic results at
  widths 1, 8, 16, and 32 across all three portable engines. Final evidence is
  HIP-off Release `19/19` in `2.02` seconds, including
  `bfnew.compact_paths` `1/1` in `0.10` seconds; ASan+UBSan `19/19` in
  `2.82` seconds, including `bfnew.compact_paths` `1/1` in `0.16` seconds.
  Strict host source/public-header syntax and fake-`__HIPCC__` production, CLI,
  and deferred-test syntax passed warning-clean.
- HIP compilation, device execution, transfer traces, device correctness,
  latency, throughput, stage timing, profiler evidence, and the statistically
  documented unbounded-Dijkstra quality sample remain deferred to
  `docs/PHASE18_GPU_VALIDATION.md`.
- Phase 18 stops before congestion or resource-conflict logic and adaptive
  hybrid selection. Phase 19 now supplies the separate final audit.

## Phase 19 final-audit and recommendation contract

- The Phase 19 report has canonical, exhaustive feature, comparison, question,
  profiler-metric, and blocker inventories: 56 feature rows, 27 comparison
  rows, 10 profiler-metric rows, 7 question rows, and 15 local blocker rows.
  Every fixed ID appears exactly once and in canonical order. Missing,
  duplicate, reordered, or unknown IDs fail validation; unknown enum encodings
  never become a fallback value.
- Each feature has exactly one of four classifications: implemented and
  correctness-tested; implemented and performance-profiled; implemented but
  not yet representative; or designed but deferred. Host correctness and
  source-level HIP checks are distinct scopes. They may not be combined to
  claim device correctness or device performance.
- An implemented-and-performance-profiled SSSP classification requires one
  representative workload/configuration identity with complete device
  correctness, ordinary unprofiled timing, and the applicable normalized trace
  or compatible PMC evidence. Bounded host duration, analytical bytes,
  portable work counters, fake-HIP syntax, and an unexecuted profiler harness
  do not qualify. The canonical local report contains zero such features.
- Every unavailable comparison or question has no numeric value, selected
  configuration, or evidence provenance. A measured conclusion requires all
  representative gates and the evidence types appropriate to that conclusion.
  Missing evidence is unavailable, not measured zero.
- The required comparison inventory covers the overall best engine; the best
  control separately for Jacobi, dense chaotic, and frontier engines; best
  ordinary-kernel `K`
  separately for all three; best persistent-grid policy separately for all
  three; batching width; overlap-planner thresholds; expansion schedule;
  P50/P95/P99 latency; all-query throughput; cold and warm end-to-end time;
  box-miss/expansion/fallback rates; path quality; resident/workspace/compact
  memory; synchronization/copy counts; and profiler-supported bottlenecks.
- The seven Phase 19 questions ask whether Jacobi min-plus pull benefits from
  the GPU; whether eliminating distance atomics outweighs dense edge scans;
  whether chaotic propagation reduces enough rounds to justify atomics;
  whether frontier work reduction offsets queue overhead; whether cooperative
  persistence beats chunking; whether overlapping batching is necessary for
  adequate utilization; and which tail-query class dominates all-query
  runtime. These are distinct from the seven Phase 12 shootout questions.
- The profiler inventory contains marker-delimited GPU-active time and PMC
  fields for L2 hit percentage, L2 read/write bytes, occupancy, memory-unit
  busy percentage, waves, and vector/scalar/memory instruction counts. Actual
  local values remain unavailable because no accepted trace or compatible PMC
  campaign exists. In a complete measured report, every profiler row uses the
  same configuration fingerprint as the profiler-bottleneck comparison, which
  must in turn match the best-engine selection.
- A production configuration is optional and absent from the local report. If
  supplied later, it must identify one legal engine/control/block/grid/chunk,
  frontier-queue capacity, batch-width, planner-fraction, expansion-schedule,
  terminal, and compact-reconstruction configuration. Frontier uses a nonzero
  capacity and non-frontier engines use zero. The shared run, planner, and
  expansion option validators plus the Phase 19 mode/fingerprint checks must
  all pass, and all toggles remain configurable. An absent unique-winner gate
  or incomplete candidate rectangle yields no recommendation.
- Supporting configuration fingerprints on a measured answer must be nonzero,
  strictly increasing, and unique within that answer. The answer's evidence
  fingerprint binds those references to the same evidence snapshot. The
  report carries only a normalized candidate-catalog fingerprint and a
  complete-matrix attestation supplied by a trusted normalized campaign;
  validation does not reconstruct raw candidate membership or recompute
  rankings from raw timing rows.
- The local blocker set is exact and canonical: missing versioned
  `logicnets_jscl` query artifact; HIP compiler validation; GPU device
  validation; comparable CPU baseline; representative correctness;
  representative timing; dependency trace; compatible PMC evidence; complete
  candidate matrix; all-query evidence; normalized per-query/class tail
  attribution; unique winner; memory evidence; synchronization/copy evidence;
  and path-quality evidence. A blocker cannot coexist with a measured
  conclusion that depends on it.
- `logicnets_jscl_unrouted.phys` and `logicnets_jscl.netlist` exist below the
  configured read-only FPGA24 root. They are inputs to deterministic artifact
  generation, not substitutes for
  `logicnets_jscl.padding1.v1.bfqueries`, which is absent locally.
- Deterministic serialization/deserialization preserves the complete validated
  report byte for byte. Malformed fields, duplicate supporting configuration
  references, trailing or unknown records, noncanonical ordering, inconsistent
  fingerprints/counts, nonfinite or signed-negative-zero numeric encodings,
  premature class-2 features, measured conclusions, or recommendations fail
  closed.
- The local result is `insufficient_evidence`: all performance comparisons and
  seven questions are unavailable, no production configuration or evidence
  references are present, and the adaptive-hybrid experiment decision remains
  deferred until representative standalone evidence exists. No hybrid is
  implemented.
- Phase 19 adds no congestion/resource ownership, historical-cost update,
  disjoint batching, exit certificate, adaptive engine, or Phase 20 work. The
  final evidence ledger and exact deferred gates are documented in
  `docs/PHASE19_FINAL_AUDIT.md`.

## Development process

- Do not use a browser or access `sjc.aupcloud` for project execution. The
  coding agent is not expected to interface with AMD GPUs directly.
- A phase passes when its implementation is complete and the full unchanged CPU
  suite passes. GPU build, execution, profiler, and measurement runs are
  optional maintainer-run evidence and are not acceptance blockers. Under the
  current testing policy, every such run, including bounded-real and full-graph
  validation, is deferred until all implementation phases are complete. Never
  describe unexecuted HIP behavior as GPU-validated or infer GPU performance
  from CPU results.
- Before editing, read `CODEX_IMPLEMENTATION_PROMPTS.md`, this contract,
  `docs/DESIGN.md`, and `docs/IMPLEMENTED_OPTIMIZATIONS.md` when present.
- Inspect the current tree and preserve completed behavior.
- Implement exactly one requested phase and do not begin later phases.
- Keep source files focused and split them before they become catch-all files.
- Use C++20, CMake, CTest, and the C++ standard library. Testing must not
  require a downloaded framework.
- Run the phase-focused tests and then the complete CPU suite.
- Update `docs/IMPLEMENTED_OPTIMIZATIONS.md` only with functionality that
  exists and has passing evidence. Planned work must remain clearly marked as
  not implemented.
- Each phase handoff reports files changed, commands run, test results,
  measured findings, remaining limitations, and the exact next phase number.

This contract records requirements, not evidence that later-phase structures
or algorithms already exist.
