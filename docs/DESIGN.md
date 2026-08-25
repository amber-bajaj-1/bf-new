# Design

## Current status

Phase 5 implements two CPU correctness solvers and post-relaxation path
reconstruction on the immutable graph. Phase 6 resets the later campaign and
adds an optional HIP build gate and measurement tools. Phase 7 adds an
independent FPGA Interchange workload bridge, deterministic artifacts, and the
first solver-facing route-query representation. Phase 8 adds deterministic
tile-run metadata, a checked 32-bit relaxation-hot graph staging layout, the
common fixed-width GPU API, and retained workspace planning. Phase 9 adds the
standalone synchronous Jacobi CSC-pull implementation: a portable host
semantic/control model and a HIP-gated kernel/driver source. Only the portable
model and source-level invariants are CPU-tested locally; the HIP source has
not been compiled or executed. Phase 10 adds the standalone dense chaotic CSR
push engine with the same portable/HIP-gated evidence split. Phase 11 adds the
standalone active-frontier CSR-push engine, including its bounded double queue,
generation deduplication, explicit overflow stop, portable controller model,
and HIP-gated source. None of the three HIP engines has local device-correctness,
occupancy, profiler, timing, or performance evidence. Phase 12 adds the
host-side controlled-shootout evidence model: canonical configurations,
fingerprinted stratified manifests, deterministic interleaving, stage and
correctness gates, timing/counter/profiler records, summaries, and deterministic
serialization. The real `logicnets_jscl` query artifact and every HIP,
representative-corpus, timing, profiler, conclusion, and default-selection
result remain uncollected. Phase 13 provides deterministic width-32/16/8
overlapping-batch plans, exact batch-device
descriptions, retained-mask and compact-descriptor run images, and checked
workspace comparisons. Its full-vertex/retained-mask decision is provisional
and supported only by bounded synthetic measurements plus an analytical
recorded-vertex-count capacity check. Phase 14 adds a bounded portable model
for overlapping Jacobi pull at widths 1/8/16/32 plus a separately HIP-gated
batch engine and retained workspace. The local evidence exercises semantics
and source-level contracts only; no HIP compiler, GPU, profiler, device timing,
or throughput result validates the device path. Phase 15 adds overlapping
batched dense chaotic push at widths 1/8/16/32, with a bounded portable model
and separately gated `BatchedDenseChaoticPushEngine`/
`ReusableBatchedDenseWorkspace` source. No local HIP compiler or GPU has
validated that source. The real `logicnets_jscl` query artifact, device
correctness, register pressure, occupancy, L2 traffic, write stalls, latency,
throughput, and uniform-versus-broadcast evidence remain unavailable. Phase 16
adds overlapping batched
active-frontier push at widths 1/8/16/32, with a bounded portable model and the
separately gated `BatchedFrontierPushEngine`/
`ReusableBatchedFrontierWorkspace` source using vertex queues and per-vertex
lane masks. No local HIP compiler or GPU validates that source, and
the generic Phase 13 one-slot estimate did not include its queues or activity
masks. Phase 17 now composes the three batch engines through deterministic
reachability-triggered expansion/replanning and compact all-query status. No
local HIP compiler, GPU, or real query artifact validates that new boundary.
Phase 18 adds generation-bound compact path reconstruction and a checked
no-congestion result/timing boundary. Phase 19 adds a canonical final-audit
report above the completed implementation; it adds no algorithm. The local
report classifies portable correctness separately from unvalidated HIP source,
leaves every representative comparison unavailable, and emits no production
configuration.

## Implemented Phase 1 representation

- `VertexId`, `TileId`, and `QueryId` wrap 32-bit unsigned values; `EdgeId`
  wraps a 64-bit unsigned value. Checked integral conversion rejects values
  outside each representation.
- `ResourceClassId` is a compact 16-bit strong ID.
- `PhysicalProvenance` is a lexicographically comparable, trivially copyable
  16-byte record. Synthetic fixtures use centralized numeric domain and kind
  constants plus deterministic source-record numbers.
- `VertexMetadata` stores an explicit coordinate-presence flag. Missing
  coordinates are canonicalized to zero-valued storage without being treated
  as valid coordinates.
- `EdgeInputRecord` stores directed endpoints, one float weight, and physical
  provenance. Accepted signed zero is normalized before validation; nonfinite,
  negative, and out-of-range inputs are rejected.
- `InputGraph` owns immutable spans of vertex and edge input records. Its edge
  order remains input-level data and carries no logical edge-ID or layout
  meaning.

## Implemented Phase 2 dual graph

- Logical edge IDs index a canonical logical-edge order keyed by original
  source ID, original destination ID, canonical float weight bits, and the
  three physical-provenance fields. Exact duplicate keys receive consecutive
  implicit parallel ranks. IDs are assigned before either sparse layout is
  sorted.
- Outgoing rows are sorted by destination ID and then logical edge ID. Incoming
  columns are sorted by source ID and then logical edge ID. Vertex IDs are
  still original IDs because spatial permutation is deferred to Phase 3.
- CSR and CSC own separate float-weight arrays. Logical edge IDs, rather than
  array positions, associate entries across the two layouts.
- Physical provenance is retained once in logical-edge-ID order as cold host
  metadata; it is not duplicated into the hot sparse views.
- The deep validator checks offset shape and monotonicity, field lengths,
  endpoint ranges, canonical zero, per-row/per-column order, one occurrence of
  every logical ID in each layout, exact transposed endpoints and weight bits,
  and canonical logical-ID order.
- Offsets and edge counts are 64-bit. The offset validator can exercise counts
  above `UINT32_MAX` without allocating a correspondingly large graph.

## Implemented Phase 3 spatial ordering

- `VertexMetadata.x/y` are the best available representative or endpoint
  coordinates. `has_location` is the sole validity indicator; no coordinate
  value is a sentinel.
- A preprocessing configuration supplies signed 32-bit origins and positive
  32-bit tile width and height. Tile division, lower-origin derivation, and
  local-coordinate subtraction use checked signed 64-bit arithmetic.
- Located tile coordinates use mathematical floor division. Unique coordinates
  receive dense `TileId` values in lexicographic `(tile_y, tile_x)` order. One
  spill tile follows all located tiles even when its vertex span is empty.
- Located vertices use the exact key `(tile_y, tile_x, resource_class,
  locality_key, original_vertex_id)`. The implemented Morton policy interleaves
  nonnegative local x/y bits into a 64-bit key.
- Some source formats may provide only one tile-level coordinate shared by
  several vertices. In that case the available local coordinates and Morton
  keys are intentionally identical. No geometry is invented; resource class
  and original vertex ID provide the remaining deterministic order.
- Spill vertices have no synthetic locality key and use only
  `(resource_class, original_vertex_id)`.
- The locality-key callable is injected only during preprocessing. Morton is
  the sole supplied policy, while another callable such as Hilbert can be
  evaluated later without adding a virtual operation to relaxation code.
- The graph stores `old_to_new`, `new_to_old`, `original_vertex_id`, per-vertex
  owner tiles, the dense tile-coordinate table, and contiguous offsets for all
  tile vertex spans.
- Stable logical edge IDs remain based on original endpoints. Rebuilt CSR rows
  sort by destination tile, destination ID, and edge ID; CSC columns sort by
  source tile, source ID, and edge ID.
- Tile-neighbor lists, edge classification, and halos are layered on by the
  Phase 4 directory rather than embedded in the Phase 3 ordering step.

## Implemented Phase 4 tile directory

- `SpatialPartitioner` is a preprocessing-time interface. The sole
  implementation is `UniformGridPartitioner`, which applies the configured
  Phase 3 ordering and constructs a validated directory. No partitioner
  virtual operation appears in a relaxation path.
- Each per-tile edge category is represented by 64-bit offsets into a flat
  stable-`EdgeId` metadata list. Internal and outgoing cross edges are owned by
  the source tile. Incoming cross entries are destination-owned references to
  those same logical edges.
- Every logical edge is classified exactly once as internal or outgoing-cross
  for its source owner. Every cross edge also appears exactly once in incoming
  metadata for its destination tile.
- Located tile neighbors are existing dense tiles at Chebyshev distance one,
  including diagonals. The lists are sorted, unique, and symmetric. A long
  located-to-located edge does not make its endpoint tiles geometric
  neighbors.
- The spill tile has no synthetic geometric neighbors. An incoming or outgoing
  cross edge incident to spill creates a symmetric spill/located adjacency.
- A tile halo is the sorted, deduplicated union of remote source and destination
  vertex IDs incident through cross edges. Halo entries resolve to the single
  global reordered vertex namespace and contain no distance values.
- Deep directory validation checks graph ownership first, then directory
  offsets, neighbor range/order/symmetry and exact policy membership, stable
  edge range/order and accounting, incoming cross completeness, and exact halo
  sets and ownership.

## Implemented Phase 5 CPU correctness algorithms

- `dijkstra_oracle` is a CPU correctness oracle for nonnegative float weights.
  `synchronous_bellman_ford` is the simple weighted min-plus reference under
  test. Neither solver contains a unit-weight branch or unweighted traversal.
- Both APIs validate, sort, and deduplicate a nonempty source set. They return
  one float label per current graph vertex plus the canonical source set.
- Relaxation stores only distance labels. Each candidate is evaluated with
  ordinary IEEE-754 float addition and accepted only by strict `<`; no
  tolerance or predecessor state participates in either algorithm.
- Bellman-Ford uses a separate next-label array for every synchronous round.
  Dijkstra uses its queue only for oracle scheduling. Both traverse the
  immutable outgoing CSR and preserve the same float candidate semantics.
- Requested paths are reconstructed afterward from final labels and incoming
  CSC edges. An incoming edge is tight only when exact float addition equals
  the current label. Tight candidates are ordered by stable `EdgeId`.
- Reconstruction uses path-local cycle detection and real backtracking, so a
  zero-weight cycle or a lower-ID tight dead end cannot trap it. The returned
  path ends at the requested target and begins at a canonical source.
- The independent path validator checks edge-ID continuity, endpoint ranges,
  simple-path termination, exact tightness, and ordinary-float accumulated
  cost against the reported and target costs.

## Floating-point verification policy

Algorithm decisions never use a tolerance. Tests built from exactly
representable values require bit-for-bit float agreement and exact path costs.
For general nonnegative data, verification may use
`nonnegative_distance_within_ulps`, whose documented default is four ULPs.
Matching infinities compare equal; NaNs and negative values are rejected by
that helper. This comparison utility is for reporting and cross-checking only,
not relaxation or tight-edge selection.

## Replacement GPU campaign architecture

The campaign defines three standalone engines with no runtime hybridization:
synchronous Jacobi min-plus CSC pull, dense in-place chaotic CSR push, and
active-frontier/worklist CSR push. Phase 9 implements the first only. Every
engine uses the common public selection/result vocabulary while retaining
distinct kernel semantics and independently supporting persistent cooperative,
chunked-host-poll, and per-round-host-poll control.

Phase 8 defines one explicit fixed-width, trivially copyable controller record.
It carries valid, active, changed, converged, execute, and next-frontier lane
masks; completed and maximum rounds; distance and frontier slot identities;
frontier sizes; done, stop-reason, and error state; engine identity; and the
per-lane convergence flag. Phase 9 Jacobi now uses a shared transition for its
portable and device paths, so executed rounds and controller slot swaps—not a
host-inferred chunk parity—determine its final buffer identity. The two push
engines will provide their own engine-specific transitions in later phases.

The Phase 3 CSR/CSC edge ordering now has deterministic maximal tile runs. A
CSR run groups one row's consecutive edges by destination tile; a CSC run
groups one column's consecutive edges by source tile. Runs never cross a row or
column boundary. Host run offsets remain 64-bit. A future batched kernel can
compute the admitted lane mask once per run and reuse it for all edges in that
run.

After all standalone engines are validated, overlapping wave32 multi-query
batching will be implemented separately for each engine. Wave64 is only a
two-wave experiment. Adaptive switching and disjoint-region batching are no
longer goals.

Queries execute on the induced subgraph of their selected tiles. Dijkstra on
that same induced subgraph is the correctness oracle; no global exit-edge
certificate is required. Unreachable lanes finish their current batch, are
collected compactly, expanded together, deterministically replanned into new
overlapping batches, and restarted from their original source sets. Reached
targets are accepted even if a cheaper path exists outside the selected region.

The Phase 7 real-workload bridge is independently implemented from the public
FPGA Interchange schemas and documented physical-netlist semantics. Historical
implementation text, tests, constants, and controller/GPU logic remain
prohibited reuse. Costs remain immutable throughout one query or batch.
Congestion is deferred to a future batch-boundary design.

## Implemented Phase 7 FPGA workload bridge

- `BFNEW_ENABLE_FPGAIF` is optional and defaults off. When enabled, CMake
  requires Cap'n Proto and generates schema sources only below the build tree.
- Large gzip-wrapped messages are decompressed into an operating-system
  temporary file and read through a private memory map, avoiding a second
  multi-gigabyte in-memory message copy.
- Device nodes become routing vertices. A node's representative coordinate is
  the lexicographically minimum `(row,col)` of its member wires, and its compact
  resource class is the minimum represented wire type.
- Tile-type PIPs are instantiated per tile. Directional PIPs emit
  `wire0 -> wire1`; bidirectional PIPs emit two distinct directed records.
  Stable provenance stores the tile ordinal, tile-local PIP ordinal, and
  direction. PIPs without two mapped nodes are counted and skipped.
- A valid schema internal delay becomes the immutable edge weight. The supplied
  xcvu3p resource lacks a complete compatible timing table, so the deterministic
  nonnegative fallback uses endpoint span, resource-class transition, and
  pseudo-PIP status in exact increments of `1/64`. It is a routing-cost policy,
  not a timing estimate.
- Primary and alternate site pins resolve through the schema parent-pin maps to
  tile wires and routing nodes. Unmappable entries are counted and omitted.
- Physical-net ingestion retains every stub terminal and one or two sources,
  while excluding `GLOBAL_USEDNET`, static, driverless, targetless, unsupported
  partially routed, unsupported-source-count, and unmappable nets.
- `RouteQuery` preserves input terminal positions and maps them to canonical
  sorted/deduplicated solver sets. Its initial selected region is the padded
  terminal-tile rectangle plus spill when needed. Deep validation and a bounded
  induced-graph helper support CPU Dijkstra checks.
- `BFGRPH07` and `BFQRY007` are explicit little-endian, versioned artifacts.
  Query records retain site/pin names, original and reordered IDs, coordinates,
  and owner tiles. Readers validate round trips; repeated files are compared
  byte-for-byte.
- `bfnew_build_fpga_workload` implements all-13 metadata scanning, complete
  `logicnets_jscl` query generation, `vtr_mcml` validation, required workload
  distributions, endpoint samples, and bounded real-query Dijkstra.

## Phase 6 HIP gate

`BFNEW_ENABLE_HIP` defaults to `OFF`, so CPU-only hosts do not need ROCm. When
enabled, CMake requires a HIP-capable CMake version and compiler and builds the
two Phase 6 tools plus the Phase 8 `bfnew_hip` runtime library. With testing
enabled it also builds the Phase 8 device-transfer CTest.

- `bfnew_gpu_probe` reports architecture, wave size, HIP scheduling-unit count,
  physical CU count when the runtime exposes it, cooperative capability and an
  executed grid synchronization, kernel-specific occupancy, runtime/profiler
  versions, accepted PMC groups, and an optional repeated profiling workload.
- `bfnew_gpu_barrier_benchmark` measures cooperative launches for legal block
  sizes 128/256/512, requested residency 1/2/4/kernel maximum blocks per WGP,
  and several grid-barrier counts. Every configuration queries its actual
  kernel occupancy.

`tools/run_phase6_profiler.sh` records the counter catalog, recursively splits
the required candidate groups until accepted, performs one real PMC collection
on the repeated workload, captures a dependency trace, and runs the barrier
matrix. These tools measure the Phase 6 gate only and implement no SSSP.

## Implemented Phase 8 shared device foundation

- `TileRunLayout64` records maximal CSR destination-tile and CSC source-tile
  runs while retaining 64-bit host offsets. Its deep validator checks bucket
  boundaries, complete edge coverage, nonempty and maximal runs, tile range and
  endpoint ownership, including empty buckets, parallel edges, long cross-tile
  edges, and spill-tile runs.
- `DeviceGraphLayout32` is a checked host staging image for the immutable hot
  device graph. It contains owner tiles, both sparse orientations, duplicated
  float weights, and both tile-run orientations. Stable edge IDs, provenance,
  and vertex metadata are deliberately absent. Construction rejects any graph,
  edge offset, or run count that cannot be represented in 32 bits; the source
  `WeightedGraph` remains unchanged and 64-bit.
- `DeviceGraphMemoryReport` reports every resident graph array by component and
  totals those byte counts before device allocation. Deep comparison checks
  integer fields exactly and float fields by bit pattern.
- Per-tile wave32 lane masks can be materialized into retained CSR/CSC run-mask
  vectors. A proof helper checks every sparse position and verifies that
  run-level owner-tile intersection admits exactly the same lane/edge pairs as
  endpoint-by-endpoint admission.
- The common API fixes the three engine kinds, the persistent-cooperative,
  chunked-host-poll, and per-round-host-poll modes, occupancy-derived versus
  explicit grid policy, instrumentation levels, stop reasons, run options,
  controller, status, work counters, and result records. The records crossing
  the future host/device boundary use fixed-width fields and are trivially
  copyable. Every result encodes the engine and control mode that produced it.
  Work statistics separately count controller copies and host synchronizations
  in addition to host checks and kernel dispatches.
- Controller and status validation enforce terminal-state and error semantics,
  not just field ranges. Done controllers have a stop reason and no pending
  changed/next-frontier mask; convergence clears active lanes and covers every
  valid lane; maximum-round termination occurs at the declared limit; and each
  error stop reason requires its exact known error bit. Status masks stay
  within valid lanes, reached and bounding-box-miss masks cannot overlap, and
  device errors cannot be reported as bounding-box misses.
- Static host/HIP ABI checks fix `GpuRunOptions` at 48 bytes,
  `DeviceController` at 96, and `DeviceRunStatus` at 48. Phase 11's appended
  frontier counters make the current `DeviceWorkStatistics` 168 bytes and
  `GpuSsspResult` 224 bytes. Every record remains eight-byte aligned.
- Workspace estimation reports source, target, selected-tile, tile-mask,
  run-mask, controller, status, instrumentation, and engine-scratch bytes
  before reservation. It reports device allocation bytes, pinned-host staging
  bytes, and their checked combined total separately. Retained capacities grow
  geometrically. CSR and CSC run masks share one allocation sized to the larger
  orientation, and all engines share only one large scratch capacity sized to
  the largest active request. Query generations and engine identity invalidate
  stale leases, and every query requires its active prefixes to be cleared.
- The optional `bfnew_hip` library provides checked HIP errors, move-only
  stream/event/device-buffer ownership, asynchronous transfers and clears,
  GPU-event timing, and steady-clock wall timing without exposing HIP headers
  through the CPU-only core library. Its device CSR/CSC/workspace views contain
  only fixed-width scalars and device pointers.
- `ResidentDeviceGraph` consumes a fully validated plan with its memory report.
  Plan validation independently checks the raw 32-bit CSR/CSC run coverage,
  bucket boundaries, tile order/maximality, and endpoint ownership before any
  allocation. The owner makes exact persistent component allocations, accepts
  only one upload attempt, retains the full host layout while the upload is in
  flight, releases that staging once readiness is observed, and exposes an
  immutable pointer view. Query preparation has no access to this owner and
  therefore cannot upload or allocate graph storage.
- `ReusableDeviceWorkspace::reserve` is the sole workspace allocation path and
  grows retained device and pinned-host buffers geometrically. Subsequent
  query preparation requires sufficient capacity, uploads retained
  source/target/selected-tile and lane-mask prefixes asynchronously, clears
  reusable status/instrumentation/scratch state, and returns a generation- and
  engine-tagged lease. An engine switch reuses the single run-mask and scratch
  allocations. The lease binds to its preparation stream; runtime wait,
  result-copy, and retirement helpers reject another stream, and retirement
  synchronizes the bound stream before invalidation. Because a raw
  `DeviceWorkspaceView` cannot reveal which stream launches a consuming kernel,
  callers must also honor that binding until a future execution-registration
  API can enforce consumer ordering.
- The HIP transfer test source includes a single-thread structural validation
  kernel for the tiny fixtures. It checks raw device CSR/CSC bucket and run
  bounds, endpoint-tile ownership, maximality, and run-lane-mask admission. It
  performs no distance relaxation and is not an SSSP engine.

The CPU tests exercise only small deterministic fixtures, including the Phase
5 spatial fixture; they do not scan the full Phase 7 graph. This section does
not describe an SSSP implementation or a measured device-speed result. The
HIP library and `bfnew.device_transfer` CTest were intentionally not compiled
or executed locally because HIP is unavailable. The AUP transfer run, HIP
allocation/copy trace, and bounded-real device sample are deferred under the
maintainer's GPU-testing policy; see `docs/PHASE8_GPU_VALIDATION.md` for the
future small-fixture command.

HIP copy calls are ordered on their supplied stream. The retained query upload
path uses page-locked internal staging. By contrast, resident-graph
uploads/downloads and status/instrumentation downloads may use caller-owned or
`std::vector` pageable memory, so their `hipMemcpyAsync` calls do not guarantee
concurrent host execution unless those host endpoints are page locked. No
overlap claim is made from the API name alone.

## Implemented Phase 9 standalone Jacobi pull

### Portable semantic and controller model

- `JacobiScratchLayout` describes exactly two adjacent graph-sized float
  columns, for a device requirement of `2 * vertex_count * sizeof(float)`.
  `JacobiDistanceView` exposes one column as `const float*` and the other as
  `float*`; binding rejects an empty, undersized, aliased, or invalid-slot
  layout.
- `run_host_jacobi_pull` is a portable CPU model of the Phase 9 semantics and
  controller protocols. It consumes the checked 32-bit graph image, a deeply
  validated `RouteQuery`, exact standalone tile masks, and Phase 8 CSC run
  masks. It is test evidence for semantics and accounting, not evidence that a
  HIP compiler or GPU executed the kernels.
- Every valid source seeds both columns with exact zero and every other vertex
  with positive infinity. One loop iteration owns each admitted destination,
  starts with its complete `d_old[v]` candidate, traverses incoming CSC
  source-tile runs whose precomputed mask contains lane zero, and uses ordinary
  float addition plus strict `<`. It writes `d_next[v]` once. There are no
  predecessor writes, target checks in a round, outgoing-CSR relaxations, or
  distance atomics.
- `advance_jacobi_controller` is shared by the portable model and HIP source.
  It validates the pending transition, increments the count for an actually
  executed round, swaps read/write slots, consumes and clears the changed
  mask, and then selects convergence, exact maximum-round exhaustion, or the
  next execute mask. A terminal controller makes later queued advances
  byte-preserving no-ops. Invalid state becomes an explicit
  `invalid_controller_state` error instead of a bounding-box miss.
- A no-change round takes precedence when it lands exactly on the round limit.
  Because every admitted destination copies or improves its preceding label,
  normal convergence leaves the two complete columns bitwise identical. The
  result takes final-buffer identity from `distance_read_slot`; it never
  derives parity from a requested chunk size.
- The portable persistent model accounts for one cooperative dispatch and one
  terminal host observation. Per-round polling accounts for an initialization
  dispatch followed by one round/advance pair and one controller copy,
  synchronization, and check per completed round. Chunked polling submits
  `K = 2,4,8,16,32` pairs per host chunk; pairs after convergence execute the
  common cheap no-op path and do not increment `rounds_completed`. These
  portable controller-copy counts model semantic host observations only; they
  do not claim to count the initial and terminal transfers made by the HIP
  runtime.

### HIP-gated execution source

- `JacobiPullEngine` is the sole Phase 9 `GpuSsspEngine`. It requires a deeply
  valid spatial host graph and tile-run layout, an uploaded immutable Phase 8
  resident graph, one retained workspace, and the workspace-bound stream. It
  rejects any engine selection other than `JacobiPull`.
- The resident plan stores a deterministic two-word fingerprint over graph
  counts and every checked 32-bit owner/CSR/CSC/run component, with float
  values included by exact bits. Engine construction recomputes the same
  sequence directly from the host graph and run sources, without allocating a
  second staging layout, and rejects a same-shaped resident upload with
  different content. This catches accidental graph/engine mismatches; it is
  not a cryptographic integrity or authenticity primitive.
- The retained engine scratch holds only the two Jacobi distance columns. The
  retained shared run-mask array is sized by the Phase 8 workspace plan. Query
  preparation uploads selected-tile masks; Jacobi initialization materializes
  every active CSC run mask once as destination-owner mask intersected with
  source-tile mask. The round reuses that value for every edge in the run and
  skips a zero run in full.
- The ordinary path enqueues one initialization kernel, followed only by
  ordered `(jacobi_round_kernel, jacobi_advance_kernel)` pairs. Per-round mode
  polls after one pair. Chunked mode enqueues exactly `K` pairs without an
  intermediate host synchronization, copies the controller after the final
  advance, and repeats only if the copied controller is not done. A final
  status kernel reads actual controller parity and target reachability. The
  physical HIP controller-copy count includes the initial workspace H2D record,
  every convergence-poll D2H record, and the terminal validation D2H record.
- The persistent path launches one cooperative query kernel. Initialization,
  run-mask construction, all relaxation rounds, controller changes, and
  terminal status remain in that kernel. Destination work is grid-stride and
  gives one logical thread owner to each destination. Every block participates
  in the pre-initialization, post-initialization, post-round, post-advance, and
  terminal grid barriers. Block zero/thread zero is the only controller owner,
  and the `done` decision is observed only after the uniform post-advance
  barrier. There is no persistent convergence copy or host synchronization.
- Each thread accumulates examined edges, successful destination decreases,
  active destinations, mask operations, and a changed bit. Dynamic shared
  memory holds one record per thread. Thread zero reduces the block and issues
  at most one global `atomicOr` to the controller changed mask. This is a
  control flag update, not an atomic distance update; the distance columns
  retain destination ownership and contain no `atomicMin`/CAS path. Optional
  instrumentation also atomically accumulates the already block-reduced
  counters.
- Both ordinary and persistent grid selection query
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` on their own real kernel with
  the selected block size and reduction shared-memory requirement. Persistent
  execution also requires cooperative-launch capability. An explicit
  blocks-per-WGP request is rejected above that measured ceiling; the
  persistent grid is the legal blocks-per-WGP value times the runtime-reported
  WGP count. The Phase 6 probe's 160-block result is not reused.
- On convergence, status checks every requested target only after all labels
  stabilize. It sets either reached or bounding-box-miss lane zero. A maximum
  round or controller/device error sets neither. Both columns are downloaded
  after termination so final slot selection and the converged bitwise-column
  invariant are explicit.

### Memory, timing, and counters

- The resident graph remains uploaded once. Query-specific device memory is
  reported and reserved through the Phase 8 workspace, including sources,
  targets, selected tiles, tile/run masks, controller, status, optional
  statistics, and the `2 * vertex_count * sizeof(float)` Jacobi scratch.
  Retained capacities can be reused; Phase 9 introduces no second
  engine-sized allocation.
- GPU-event timing in the HIP source starts after stream-ordered query
  preparation and covers the device-timeline span from initialization through
  terminal status materialization. The final controller/status/statistics/two-
  column downloads are outside that interval. For host-poll controls, the
  event interval also includes stream-idle gaps while the host observes the
  controller and submits the next chunk; it is not a sum of kernel-active
  durations and must not be compared across controls as pure GPU work. Host
  wall timing starts after preparation has been enqueued and ends after result
  transfer and same-stream lease retirement, so the two timings intentionally
  have different scopes. Per-kernel profiler durations remain deferred.
- With instrumentation disabled, no device statistics record is reserved.
  Light records rounds, examined CSC edges, useful destination decreases, and
  active destinations; Debug additionally records run-mask operations.
  Host-side accounting adds submitted kernel pairs, convergence checks,
  physical controller copies, and synchronizations. For `N` ordinary
  convergence polls, controller copies are `N + 2`; persistent has zero polls
  and two copies. HIP synchronizations are `N + 1`, covering the `N` polls and
  one shared final result-transfer/lease-retirement boundary. Timing events are
  already complete after that terminal synchronization and do
  not add another physical host synchronization. Queue and distance-atomic
  counters remain zero for Jacobi. Instrumented timings are not production
  timings.

### Phase 9 validation boundary

The portable CPU matrix covers the Phase 5 core, spatial/spill, equal-path,
and zero-cycle fixtures; a 33-vertex long chain; disconnected non-targets;
parallel edges; duplicate multi-source input; a bounded route that deliberately
excludes a cheaper global path; a converged bounding-box miss; and three fixed
random seeds. Every case runs persistent, per-round, and chunked control with
`K = 2,4,8,16,32` and compares with Dijkstra on the same induced
subgraph. Representable fixtures require bitwise equality; random fixtures use
the existing four-ULP reporting policy. Additional checks cover scratch
overflow, typed nonaliasing, terminal no-ops, maximum-round status in all three
controls, full-`K` queued no-ops at the round limit, actual-round parity, exact
dispatch/copy/synchronization counts, complete-column identity,
run-admission equivalence, validator-clean error conversion for several corrupt
controller fields, deterministic/content-sensitive graph fingerprints,
None/Light/Debug counter-level behavior, and a structural audit of the HIP
source.

The HIP-only test source independently defines 13 tiny cases: the Phase 5
core and spatial/spill fixtures, two long-chain target placements,
disconnected, zero-cycle, parallel-edge, multi-source, cheaper-outside-box,
box-miss, and three seeded random graphs. It selects all three modes and all
five `K` values, and the core case requests occupancy-derived plus fixed one-
and two-blocks-per-WGP cooperative grids. It also checks wrong-engine rejection
and maximum-round parity in persistent, per-round, and `K=32` chunked control,
and it constructs a same-shaped resident upload with
one changed weight to exercise the identity guard. It passed only a host-side
fake-declaration syntax check; no HIP compiler or GPU ran it locally.
It is therefore a deferred test plan, not passing HIP evidence. No bounded-real
or full Phase 7 graph is scanned in Phase 9. See
`docs/PHASE9_GPU_VALIDATION.md` for the eventual combined GPU-campaign
commands.

## Implemented Phase 10 standalone dense chaotic push

### Portable semantics and atomic domain

`DenseScratchLayout` reserves exactly one `uint32_t` word per vertex.
`DenseDistanceView` binds that single in-place array; no second distance
column, predecessor, frontier, or queue exists. The word is the exact bit
encoding of canonical positive zero, a finite nonnegative float, or positive
infinity. The portable conversion and atomic-min model reject negative zero,
negative values, NaNs, and bit patterns above positive infinity. Tests sample
mantissa/exponent boundaries and prove that unsigned ordering matches float
ordering over the supported domain.

`run_host_dense_chaotic_push` consumes the checked 32-bit graph image, exact
standalone tile masks, and exact CSR run masks. Each full scan visits every
admitted outgoing edge. It reloads the current source label for every edge and
applies a strict unsigned-bit minimum at the destination. Forward, reverse,
and alternating portable schedules deliberately produce different progression
on the long chain while converging to identical final bits. This is a semantic
model of asynchronous/chaotic push, not an assertion that the GPU uses a
deterministic serial schedule.

`advance_dense_controller` is shared by portable and HIP code. It consumes the
block-published changed mask, counts exactly one complete executed scan,
selects convergence before the round-limit stop when both apply, and keeps
both distance slot fields at zero. Done pairs are byte-preserving no-ops.
Per-round control submits one pair per observation; chunked control submits a
complete `K`-pair chunk even when an early pair sets `done`; persistent control
is represented as one query dispatch.

### HIP-gated execution source

`DenseChaoticPushEngine` binds the same validated host graph/run source,
immutable resident graph, reusable workspace, and one stream. Its constructor
uses the resident graph's content-sensitive two-word fingerprint, not shape
alone. Query preparation selects only CSR run capacity and one graph-sized
scratch allocation.

The device round is CSR source-row grid-stride work. It intersects source and
destination owner masks once per run and skips a zero run without inspecting
its edges. Every admitted edge uses a CAS-compatible load of `d[source]` and a
CAS-loop unsigned minimum on `d[destination]`; no ordinary distance read is
mixed with those atomic writes. Thread-local work is reduced per block, so a
changed block publishes at most one controller `atomicOr`. Optional statistic
atomics are separate from the distance update protocol.

Ordinary control launches one initializer followed by exact round/advance
pairs and a final status kernel. Per-round mode copies the controller after
one pair; chunked mode copies only after all `K` queued pairs. Persistent mode
launches exactly one cooperative query kernel containing distance/source
initialization, CSR run-mask setup, all full scans, controller changes, and
terminal status. All workgroups execute uniform pre/post transition grid
barriers, including idle workgroups and the terminal round. Both launch paths
query occupancy on their actual dense kernels and dynamic shared-memory size;
fixed blocks per WGP are allowed only at or below that result.

Light counters include admitted edges, decreases, active sources, executed
full-edge rounds, and active-lane rounds. Debug additionally counts CSR mask
operations, attempted/successful distance atomics, changed-block publications,
and unique selected destinations with at least two admitted incoming edges.
The latter is classified once during initialization from immutable CSC runs;
CSC is not part of the relaxation round. Queue/frontier counters remain zero.
Physical HIP controller copies are initial H2D plus `N` ordinary polls plus
terminal D2H; synchronizations are `N + 1`: the polls plus one shared final
result-transfer/lease-retirement boundary. Timing-event elapsed queries do
not add a synchronization. As in Phase 9, the event interval for host-poll
modes includes stream-idle re-enqueue gaps and is not summed kernel-active time.

### Phase 10 validation boundary

The bounded CPU test runs the 13 Phase 9 cases plus high-fan-in and hub-fan-out
fixtures through persistent, per-round, and chunked `K = 2,4,8,16,32`
protocols. It checks bounded Dijkstra agreement, bitwise equality for exact
fixtures, the four-ULP policy for seeded general weights, control-invariant
final bits, exact source zeros and run admission, three adversarial host
schedules, repeated execution, maximum-round behavior in all controls, full
`K=32` queued no-ops, None/Light/Debug separation, and no frontier accounting.
It also performs a structural scan of the HIP source for CSR traversal,
atomic-compatible loads/minima, occupancy-derived cooperative launch, uniform
grid synchronization, and absence of frontier/worklist code in the round.

The HIP-only test source repeats the tiny matrix, uses bounded Dijkstra as its
oracle, requests multiple runtime-legal cooperative grids, checks repeated
final-bit stability and instrumentation, and covers maximum-round exhaustion
for all controls. It has received only a host-side declaration syntax check;
no HIP compiler or GPU has run it. See `docs/PHASE10_GPU_VALIDATION.md` for the
deferred commands. No bounded-real or full Phase 7 graph is part of Phase 10.

## Implemented Phase 11 standalone active-frontier push

### Scratch, queue, and relaxation semantics

`FrontierScratchLayout` describes one graph-sized 32-bit distance-word array,
two bounded 32-bit vertex queues, and one graph-sized 64-bit enqueue-generation
array. A zero requested queue capacity selects `vertex_count`; a nonzero
capacity must be in `[1, vertex_count]`. The layout aligns the generation array
to eight bytes. For `V` vertices and queue capacity `Q`, its bytes are
`4V + 8Q + alignment_padding + 8V`; the default `Q = V` is therefore `20V`
plus at most four alignment bytes. It checks every size/offset calculation and
binds only a sufficient, aligned, nonoverlapping allocation. The default
capacity cannot overflow when generation deduplication is correct. An
explicitly smaller capacity exists to validate overflow behavior.

The initial read queue is the complete canonical source set: sorted,
deduplicated sources appear once, with exact zero labels, while every other
distance word begins at positive infinity. Empty initial frontiers are rejected.
All sources and targets remain one multi-source/multi-target query rather than
separate searches; target reachability is classified only after convergence.
Every graph-sized enqueue-generation word is reset to zero during this query
initialization, so the first round's generation value of one is unambiguous.
Each current-frontier entry is processed by one grid-stride thread, which owns
that source's outgoing CSR row. Source/destination tile admission is
materialized once per immutable CSR destination-tile run; a zero-mask run is
skipped without edge inspection. The engine does not traverse CSC and has no
prefix-sum edge balancing, virtual-warp/high-degree path, wave aggregation, or
adaptive scheduler.

This deliberately simple mapping can expose outgoing-degree imbalance because
one queue entry's thread scans its whole admitted row while another may have no
edges. High fan-in can contend on destination minima and generation exchanges;
all unique admissions also contend on the next-size reservation and publish
the standalone next-frontier mask. Per-block reduction limits statistics
atomics but does not remove those algorithmic atomics. These are mechanisms to
measure later, not evidence that any one is a bottleneck.

Every admitted edge performs a CAS-domain source load and strict unsigned-bit
minimum on its destination. On a successful decrease, a 64-bit atomic exchange
sets the destination's enqueue generation to `rounds_completed + 1`. Only the
first exchange that observes another generation atomically reserves a next
queue slot. A later successful improvement in the same round still updates the
label but records a duplicate suppression and does not append another entry.
The reservation increments the actual next size even at capacity; a claim
outside the bounded queue sets `queue_overflow` and performs no queue write.
Overflow is therefore explicit and cannot silently truncate the frontier.

`advance_frontier_controller` is shared by the portable model and HIP source.
After a valid executed round it increments the round count, swaps queue slots,
and clears the recycled old-read size. It then chooses queue overflow/device
failure, empty-next-frontier convergence, exact maximum-round exhaustion, or
continuation. Convergence wins when the final allowed round also produces an
empty next frontier. Malformed state becomes a validator-clean terminal error;
already-terminal advances are byte-preserving no-ops. The single in-place
distance array keeps both controller distance slots and final status slot at
zero. Reached or bounding-box-miss state is computed only after normal
convergence; limits and errors are not misses.

### Three controls and HIP-gated source

`run_host_frontier_push` models persistent cooperative, per-round host poll,
and chunked host poll with `K = 2,4,8,16,32`. Per-round control submits one
round/advance pair per observation. Chunked control always submits all `K`
pairs, even if an early pair reaches `done`; later pairs no-op and cannot alter
the true completed-round count or queue parity. Its accounting represents
semantic host observations, not physical HIP transfers.

The HIP-gated `FrontierPushEngine` binds a deeply validated host graph/run
image, immutable resident upload, reusable workspace, and one stream. In
addition to shape checks, construction compares the resident graph's
deterministic two-word fingerprint with the exact host graph/run source. This
is an accidental same-image guard, not a cryptographic guarantee. Query
preparation requests the Phase 11 scratch layout and CSR run-mask capacity;
the graph remains resident across runs.

Ordinary HIP control has one initializer, ordered
`(frontier_round_kernel, frontier_advance_kernel)` pairs, a controller poll
after each one- or `K`-pair chunk, and one final-status kernel. For `N` polls,
physical controller copies are initial H2D plus `N` D2H observations plus the
terminal D2H (`N + 2`), while host synchronizations are `N + 1`: the polls
plus one shared final result-transfer/lease-retirement boundary.
Timing-event elapsed queries add no synchronization. Persistent control instead
launches exactly one cooperative query kernel containing
initialization, run-mask construction, queue rounds, controller transitions,
and final status. All workgroups cross uniform grid barriers around every
transition, and no frontier-size or convergence value is polled by the host
inside that kernel.

Both ordinary and persistent paths query occupancy on their real frontier
kernels with the selected block and block-reduction shared-memory size. Fixed
blocks per WGP are accepted only when no larger than the real result. The
ordinary grid is additionally capped by queue-capacity work blocks; the
cooperative grid remains residency-bounded so every block can participate in
every grid barrier. No Phase 6 probe limit is reused.

### Counters, timing, and result contract

The resident graph is uploaded once and reused. Per-query workspace planning
reports sources, targets, selected tiles, tile and CSR run masks, controller,
status, optional statistics, and the checked frontier scratch allocation.
Capacities are retained across compatible runs, and Phase 11 introduces no
second engine-sized workspace allocation outside that shared scratch.

None allocates no device statistics record and suppresses frontier algorithm
counters. Light records admitted edges, successful decreases, active frontier
vertices, executed frontier rounds, maximum queue size, and empty/small
frontier rounds. The portable `current_frontier_sizes` vector is the exact
per-round sequence and is checked against the aggregate. The HIP record exposes
`active_vertices` plus `active_lane_rounds` as aggregate per-round accounting;
it does not claim or download a device per-round trace. Debug adds CSR run-mask
operations, atomic attempts and successful updates, queue claims, duplicate
suppressions, and explicit overflow events. Control counts are populated
independently of algorithm instrumentation. Expansion and dense-only
high-contention, full-edge, and changed-publication counters remain zero.

`DeviceWorkStatistics` is now 168 bytes and `GpuSsspResult` is 224 bytes, both
eight-byte aligned. The result retains frontier engine/control identity,
terminal status, and work counters; the enclosing frontier run output retains
the final controller/queue parity and timing metrics. Its HIP event begins
after stream-ordered query preparation and spans initialization through
terminal status. Final
controller/status/statistics/distance downloads are outside that event. In
host-poll modes `gpu_milliseconds` also includes stream-idle gaps while the host
observes and re-enqueues work; it is not summed kernel-active time and must not
be compared across controls as pure GPU work. Host wall timing begins after
preparation has been enqueued and ends after result transfer and same-stream
lease retirement. Separate profiler kernel durations remain deferred evidence.

### Phase 11 validation boundary

The bounded CPU test runs 17 tiny fixtures: the 13 Jacobi cases, both dense
contention/shape cases, an expanding grid, and a repeated-improvement case.
This includes sparse-chain, high-fan-in, hub-fan-out, zero-cycle,
multiple-source, disconnected, and bounded-miss behavior. It runs persistent, per-round, and
chunked `K = 2,4,8,16,32`, compares with bounded Dijkstra, requires bitwise
equality for exact fixtures and the established four-ULP policy for seeded
general weights, and requires
control-invariant final bits. Focused assertions cover exact scratch offsets,
alignment/capacity rejection, canonical source initialization, queue-slot
parity, sparse and expanding frontier shapes, generation deduplication,
explicit overflow in every control, maximum-round/error/miss separation,
None/Light/Debug counters, repeat stability, and absence of deferred schedulers
from the HIP round source.

Bounded CPU evidence is 11/11 CTests in `build/phase11-cpu-release` from
`ctest --test-dir build/phase11-cpu-release --output-on-failure` (0.09
seconds), including `bfnew.frontier_push`. The focused binary reports a maximum
seeded-random difference of 0 ULP.

The HIP-only `bfnew.frontier_push_gpu` CTest source repeats the 17-fixture
control matrix and separately covers None/Light/Debug, occupancy-derived and
fixed one-/two-blocks-per-WGP cooperative grids, maximum-round exhaustion,
capacity-one overflow, and empty-source rejection. It is wired into the
optional HIP build. The final HIP engine source passed a host C++20 fake-HIP
declaration syntax check with only the expected fake-stub warnings, and the HIP
test source passed its warning-clean host C++20 syntax check. These remain
source-level checks: neither source has been compiled with a HIP compiler or
executed on a GPU. No device correctness, real occupancy, dependency trace,
PMC, isolated timing, bounded-real, or full-graph evidence exists. All such
runs remain deferred until the combined GPU campaign after all implementation
phases; exact eventual instructions are in
`docs/PHASE11_GPU_VALIDATION.md`.

## Implemented Phase 12 controlled single-query shootout evidence layer

### Tuning and runtime-legality catalog

`engine_shootout.hpp` defines a schema-versioned host representation for the
single-query comparison without adding a relaxation engine. A
`ShootoutTuning` names exactly one of Jacobi pull, dense chaotic push, or
frontier push and exactly one of persistent cooperative, per-round host-poll,
or chunked host-poll control. `make_shootout_tunings` emits block sizes 128,
256, and 512 for every engine. Chunked entries cover `K = 2,4,8,16,32`;
persistent entries include an occupancy-derived choice and every caller-supplied
fixed blocks-per-WGP choice. Instrumentation is deliberately a run-stage
property rather than tuning identity, allowing timing and work records to join
on one configuration ID without mixing their instrumentation levels.

`resolve_shootout_configurations` combines that catalog with per-engine,
per-block runtime kernel limits. It preserves a decision for every entry and
distinguishes a missing occupancy row, illegal block size, invalid tuning, and
fixed cooperative residency above the real kernel limit. Thus an unavailable
or illegal configuration cannot disappear from the matrix silently, and the
Phase 6 probe's 160-block result is never treated as an engine limit.

### Fingerprinted representative manifest

`ShootoutInputFingerprint` binds two graph words, two query-corpus words, the
complete corpus query count, and schema version. The fingerprint is an
accidental campaign-identity guard, not a cryptographic authenticity claim.
Every manifest and sample repeats it, and validation rejects evidence from a
different graph, query corpus, or schema.

Candidate metadata records stable query ID, selected-region vertex and edge
counts, fanout, source count, and expected rounds. Expected rounds has one
fixed definition: the completed-round count from a deterministic Jacobi pilot.
The real pilot first computes the four cheap features across eligible metadata,
uses a domain-separated seed to select a stratified shortlist no larger than
four times the requested count, and runs status-only Jacobi only for that
shortlist. It then performs final five-dimensional selection. Pilot execution
exists only to stratify workload shape and is never a timing sample.
`select_logicnets_shootout_manifest` requires at least 1,000 requested entries,
rejects duplicate or malformed candidates, and places all five dimensions into
deterministic four-way quantile bins. It apportions requested rows across
occupied combined strata proportionally with deterministic largest-remainder
allocation; a seed-keyed stable hash chooses within each stratum. The serialized
manifest retains the seed, features, bins, and selection order.

The real `logicnets_jscl` artifact is not currently present. The existing
`out/phase7/phase7_partial_report.v1.txt` says the full real graph artifact was
created but the all-13 scan and query-artifact emission were stopped. The CPU
tests therefore use small and synthetic metadata, including an at-least-1,000-
candidate selector exercise; they are not representative-query execution.
The deferred GPU runner provides separate built-in `sparse-wavefront` and
`dense-frontier` workload identities. They cannot replace or be pooled with
the 1,000 real queries.

### Interleaving and evidence-stage gates

`make_interleaved_shootout_schedule` assigns every selected
query/configuration/repetition tuple exactly once and gives it a unique global
execution ordinal. Seeded query and configuration orders change by repetition,
and the configuration order is rotated across queries. This preserves a
reproducible interleaving rather than running every sample from one engine or
control contiguously. The GPU driver uses that interleaving for warmed timing,
counters, and profiler case identities. Correctness alone deterministically
groups configurations by query and reassigns truthful ordinals, ensuring that
the induced bounded-Dijkstra oracle is built exactly once per query.

`ShootoutRunKind` separates warmup, correctness, unprofiled timing, algorithm
counters, trace, and PMC samples. `validate_shootout_samples` enforces these
boundaries:

- correctness uses instrumentation None, downloads final labels, and records
  no timing;
- final timing uses instrumentation None, a status-only transfer, and complete
  measured preparation, SSSP device-timeline, result-transfer, and wall fields;
- algorithm counters use Debug, do not download graph-sized labels,
  and cannot carry timing evidence; and
- warmup, trace, and PMC records are status-only None runs and remain separate
  from ordinary timing.

Every sample also retains engine/control result identity, controller status,
work counters, actual cooperative grid and active blocks per WGP, and profiler
fields whose state is explicitly `unavailable`, `not_applicable`, or
`measured`. Missing and inapplicable PMC values are therefore never serialized
as misleading measured zeros. `require_complete_correctness_gate` requires one
passing correctness record for every manifest/configuration pair before any
matching timing, counter, trace, or PMC record is accepted.

The HIP-gated `EngineShootoutExecutor` owns one instance of each of the three
existing engines over a shared resident graph, reusable workspace, and stream.
It dispatches no other engine. Correctness calls each engine's selected-distance
path; warmup, timing, counters, and profile replay call the common status-only
path and reject any returned label vector. Timing additionally fails if the
retained workspace's allocation-event count grows across an engine call. A
`ShootoutTilePairIndex` scans immutable CSR tile runs once, then computes each
pilot query's selected-edge feature from tile-pair rows rather than rescanning
all graph vertices or edges per query. Correctness copies selected tile ranges
only and constructs one admitted-CSR-run local Dijkstra oracle per grouped
query. The Debug counter stage resolves its own kernel limits and records
configurations that become unavailable under Debug. Each batched profiler case
receives one unrecorded status-only warmup and a named begin/end marker-kernel
range in a persisted ledger.

The comparison timing boundary is intentionally fair across engine scratch
sizes. Correctness downloads selected-tile final distances for local bounded-
Dijkstra comparison outside timing. Production timing transfers only common status and
records preparation, the SSSP device-timeline span, result transfer, and
end-to-end wall separately. A host-poll device timeline includes the stream-idle
host observation/re-enqueue gaps and is not summed kernel-active time. Trace
evidence is required to discuss active kernel duration.

### Deterministic summaries and evidence boundary

`summarize_shootout_campaign` computes nearest-rank P50/P95/P99 distributions
for wall and SSSP device-timeline values, aggregate throughput from the summed
end-to-end wall boundary, rounds, examined edges, successful decreases and
their ratio, active/frontier work, atomics, queue work, dispatches, host
synchronizations, controller copies, and profiler averages. It retains up to
50 samples at or beyond each configuration's wall-time P99 together with their
five query features. Manifest, catalog, and sample TSV plus report JSON
serialization use stable ordering and locale-independent numeric formatting.

The report deliberately initializes all seven Phase 12 questions as pending:

1. persistent cooperative versus chunked control;
2. the best K for each ordinary-kernel engine;
3. cooperative occupancy sensitivity;
4. dense pull versus sparse frontier regions;
5. whether dense chaotic push is useful beyond diagnosis;
6. which locality, round, atomic, or box feature explains the tail; and
7. how much per-round polling time is eliminated.

Recommended defaults likewise remain `pending measured GPU campaign`, and all
engine, control, K, grid, and block-size toggles remain configurable. The
host-only `bfnew_shootout_report` consumes and validates serialized campaign
records and emits a stable, globally capped profile-case plan. It first covers
every available tail-metric/engine/control stratum, then fills by descending
tail severity, with at most 48 distinct cases. Every plan row repeats the input,
workload, selection-seed, manifest-order-seed, replay-order-seed, and schedule
policy identity. An optional conclusions TSV can replace the seven pending
answers and pending default only with workload-matched, configuration-matched,
supplied-evidence-backed measured or insufficient-evidence records; all toggles
must remain configurable. The host-only `bfnew_shootout_profile_import`
validates normalized trace/PMC values against either one isolated staging row
or a strict one-to-one map from stable profile-case IDs to every row in a
marker-delimited batched replay. The
`bfnew_gpu_shootout` driver is gated on both HIP and FPGA
Interchange support so the default build retains neither dependency. Merely
building either tool does not supply representative or performance evidence.

### Phase 12 validation boundary

The focused `bfnew.shootout` CPU test exercises catalog enumeration and runtime
legality decisions, the deterministic at-least-1,000-entry five-dimensional
selector, repeatable interleaving, stage and fingerprint rejection, correctness
gating, percentile/throughput/counter/long-tail summaries, explicit evidence
states, pending conclusions/defaults, and byte-stable TSV/JSON. It also uses the
existing tiny adversarial engine fixtures for bounded-Dijkstra and
control-invariant correctness where applicable. It performs no full graph
scan, real query search, HIP compilation, GPU execution, timing, trace, or PMC
collection.

CPU acceptance evidence: the Release build in `build/phase12-cpu` passed all
12/12 bounded CTests in 1.10 seconds with
`ctest --test-dir build/phase12-cpu --output-on-failure`. The focused
`bfnew.shootout` binary passed and explicitly reported that it collected no GPU
timing evidence. The ASan/UBSan build in `build/phase12-asan` passed the focused
test 1/1 in 0.31 seconds.

The deferred `gfx1151` procedure is in
`docs/PHASE12_GPU_VALIDATION.md`. Until that procedure is explicitly run after
all implementation phases, there is no HIP compiler result, device correctness,
actual-kernel occupancy, real 1,000-query manifest, wall/device timing
distribution, trace, PMC, long-tail diagnosis, performance conclusion, or
recommended default.

## Implemented Phase 13 deterministic overlapping-batch planning

### Indexed features and frozen greedy order

`SelectedRegionIndex` binds to one deeply validated spatial `WeightedGraph`
and its maximal tile-run layout. Construction walks immutable CSR runs once,
coalesces their exact edge counts by ordered source/destination tile pair, and
stores vertex counts from the graph's contiguous tile spans. A query or batch
union estimate then visits only selected tile spans and the selected source
tiles' indexed destination rows. It never performs the former per-query scan
over all graph vertices or sparse edges.

`make_batch_query_features` requires that graph identity, deeply validates each
`RouteQuery`, retains distinct canonical source and target sets, derives their
owner-tile sets, and records actual selected tiles plus exact selected vertex
and induced-edge counts. The result is sorted by the full 32-bit `QueryId`
domain. Consequently reversing input `RouteQuery` order produces the same
feature order, plan indices, lane identities, and serialized plan.

The standard family emits plans in wave32-first order: widths 32, 16, and 8.
Width 64 is intentionally absent. Planning begins from the unassigned anchor
with greatest selected-edge count, selected-vertex count, then selected-tile
count; smaller expansion generation and QueryId break remaining ties. Each next
lane is scored against the accumulated batch unions by:

1. greatest selected-tile Jaccard;
2. greatest source-tile plus target-tile overlap;
3. lowest projected union inflation, exact union edges, exact union vertices,
   and newly introduced tiles; then
4. lowest differences from the anchor in expansion generation, source count,
   and target count, followed by QueryId.

The default gates require Jaccard at least `1/8` and projected union inflation
at most `2`. Inflation is the exact rational
`valid_lane_count * union_tile_count / sum(selected_tile_count_by_lane)` after
adding the candidate. Fraction comparisons avoid overflow-prone cross products.
If no eligible candidate remains, the batch closes and the next canonical
anchor begins another. Valid lanes always form the low contiguous mask prefix;
all lane arrays retain exactly the requested width, and invalid padded lanes
have no semantic payload. Thus every input query is assigned once even when a
remainder is a singleton. The deep validator reconstructs identities, unions,
lane masks, estimates, once-only assignment, and the complete greedy result.
`serialize_batch_plan_tsv` emits stable `bfnew.batch-plan.v1` plan, batch, lane,
and tile rows.

### Device-description boundary without a batched kernel

`BatchDeviceDescription` is an owning reusable host preparation image, not a
HIP workspace or SSSP implementation. It contains:

- QueryIds and expansion generations by lane;
- independent flattened source and target arrays with per-lane offsets;
- selected vertex and edge estimates by lane;
- the sorted union tile list, a graph-tile lane-mask array, and exact selected
  vertex ranges;
- an optional reusable dense per-tile compact-index bias map, where a selected
  global vertex maps as `global_vertex - bias[owner_tile]`;
- zero-initialized reached and miss masks; and
- either retained CSR/CSC run masks with touched ledgers or sorted compact
  nonzero `(run_id,lane_mask)` descriptors with per-union-vertex CSR/CSC
  descriptor offsets.

Preparation resolves original `RouteQuery` records by QueryId, so canonical
feature ordering does not merge or reorder a lane's terminals. For a CSR run,
the stored mask is source-owner-tile lanes intersected with destination-tile
lanes. For CSC it is destination-owner-tile lanes intersected with source-tile
lanes. The value is computed once per maximal run and reused for all edges in
that run; a zero mask is skipped. Retained storage clears only runs named by
the preceding touched ledger. Compact storage emits only nonzero run IDs in
strict order. Preparation reports visited/active runs, lane-edge pairs, clears,
and writes. Cold initialization traffic and warmed clear/write traffic are
reported separately, and a reusable image is bound to the first selected run
representation so unmodeled retained/descriptor switching is rejected. Its
validator independently rebuilds payloads, ranges, mappings, descriptor
offsets, masks, run coverage, and per-lane admitted-edge totals. A bounded
proof expands compact descriptors when needed and reuses the Phase 8
endpoint-by-endpoint admission proof.

### Workspace comparison and evidence boundary

The checked model evaluates four combinations for the same batch:

- full graph-sized vertex-major lanes with retained per-run masks;
- full graph-sized vertex-major lanes with compact nonzero descriptors;
- compact union-tile lanes with retained per-run masks; and
- compact union-tile lanes with compact nonzero descriptors.

It separates distance, global/local tile mapping, run storage, descriptor
offset, and batch-metadata bytes. It also reports distance-reset traffic,
mapping and run-preparation writes, selected and wasted lane vertices,
active/zero runs, reusable allocation, and maximum concurrent workspaces under
an explicit capacity/resident/reserve budget. Every multiplication and sum is
checked before a reservation. `ReusableBatchWorkspaceReservation` retains the
largest component capacities and grows them geometrically. The max-CSR/CSC
device run-storage and write model assumes one fixed active orientation per
reused workspace; an engine that switches orientation needs separate workspace
accounting.

The acceptance fixture compares all four combinations and provisionally
selects full graph-sized vertex-major storage with retained run masks. On that
bounded fixture it avoids global/local label mapping and uses the simplest
reusable touched-run preparation. Exact fixture bytes, preparation counts, and
host build times are recorded in `docs/PHASE13_WORKSPACE_DECISION.md`. They do
not predict the real graph or a GPU kernel.

The only real-scale calculation in Phase 13 is an analytical label-array guard
using the Phase 7 record `V = 28,226,432`. Exact `V * width * slots * 4` bytes
are:

| Width | One slot | Two slots |
| ---: | ---: | ---: |
| 8 | 903,245,824 | 1,806,491,648 |
| 16 | 1,806,491,648 | 3,612,983,296 |
| 32 | 3,612,983,296 | 7,225,966,592 |

An illustrative, non-runtime scenario uses 64 GiB nominal device capacity, an
8 GiB resident-graph allowance, a 4 GiB explicit reserve, and two distance
slots. It shows only that the width-32 label component is arithmetically
representable within that assumed envelope. It does not measure free memory,
include real run/mapping/metadata bytes, establish a concurrent-workspace
count, or authorize an allocation.

The real `logicnets_jscl.padding1.v1.bfqueries` artifact was not emitted in the
interrupted Phase 7 run. Therefore real selected/union tile distributions,
active-run ratios, mapping time, reset traffic, runtime free memory, concurrent
workspace count, and GPU preparation/execution time are missing. The
full-plus-retained choice remains provisional until those are measured. Phase
13 adds no HIP source and no batched relaxation or convergence behavior.

### Phase 13 CPU validation boundary

The focused tests use only tiny deterministic graphs. They cover exact indexed
counts, the width-8 golden plan, permutation invariance, widths 32/16/8 and
singleton padding, one- and two-source retention, exact tile and run masks,
retained-ledger reuse, compact descriptor order, endpoint admission proof,
overflow/capacity rejection, four-way workspace accounting, and stable
decision serialization. The final bounded Release suite passed `14/14` tests
in `0.05` seconds. A separate focused plan/workspace invocation passed `2/2`
in `0.26` seconds, and the focused ASan+UBSan pair passed `2/2` in `0.45`
seconds.
No real artifact, full-graph scan, HIP compiler, GPU, trace, or PMC run belongs
to this evidence; the tiny host timings are structural measurements, not
performance evidence.

## Implemented Phase 14 overlapping batched Jacobi pull

### Vertex-major lane state and input identity

`batched_jacobi_pull` accepts only widths 1, 8, 16, and 32. The Phase 13
standard planner remains 32/16/8; width one is constructed explicitly for the
standalone-equivalence baseline. Each of two distance slots contains
`vertex_count * lane_width` floats, with
`index = vertex * lane_width + lane`. This preserves direct global-vertex
addressing while keeping the query lanes for one vertex contiguous.

One low-prefix validity mask distinguishes real lanes from padding. Validity
never depends on the QueryId payload because the full 32-bit QueryId domain is
legal. Source and target arrays are flattened only for transfer; independent
per-lane offsets remain semantic boundaries. Every canonical source in a lane
is written as exact positive zero in both slots. Identical source vertices in
different lanes remain distinct distance cells, and a multi-source lane seeds
all of its sources without affecting another lane.

Only lane/vertex cells admitted by that lane's selected tiles are semantic.
Initialization writes both slots for those cells to zero or positive infinity.
Padded and nonselected cells are neither relaxation inputs nor correctness
outputs. Consequently slot-parity validation and bounded-oracle comparison are
performed over each lane's selected region, not over stale unused capacity.

### Shared CSC traversal and exact per-lane relaxation

For every selected destination, the range lane mask is intersected with the
controller execute mask. Incoming edges retain the Phase 8 CSC order and
source-tile runs. The prepared run mask already represents destination-tile
admission intersected with source-tile admission; the round intersects it once
more with the currently executing lanes. A zero result skips the complete run.
For a nonzero run, each source/weight record is loaded once by the portable
model and applied independently to every set query-lane bit. Every candidate
uses ordinary float addition and strict `<`; there is no cross-lane reduction,
source merge, tolerance, or distance atomic.

The bounded portable path deeply reconstructs batch identity, terminal slices,
tile/range masks, CSC run coverage, endpoint admission, selected counts, and
per-lane edge totals before execution. It accepts retained masks and compact
nonzero descriptors and proves that either drives the same relaxation order.
That deep scan is a bounded validation path, not the intended real-corpus hot
preparation loop. The production HIP path accepts a device-materialized
description with no host run image. During initialization, the wave assigned
to each selected destination cooperatively writes every CSC run mask as the
destination-range mask intersected with the source-tile mask. It writes zeros
too, so every run the batch can read is current; stale entries outside selected
columns are never read. This removes the graph-wide clear and H2D run-mask copy
without changing the repeated round's single dense mask load. Retained host
masks remain a deep-validated compatibility path for bounded tests.

### Per-lane convergence and slot parity

A round reads one immutable slot and writes every selected destination for each
executing lane into the other slot, including an unchanged preceding value.
Only after every write completes does the shared Jacobi controller increment
the real-round count and swap the slot identities. The no-change bits are then
eligible for convergence.

With `enable_per_lane_convergence=1`, a no-change lane moves from active to
converged and performs no later semantic relaxation. Its final no-change round
has already made its selected cells bitwise equal in both slots; later global
slot swaps therefore preserve that lane without copy repair. With the flag
disabled, every valid lane continues until a batch-wide no-change round. The
first no-change round is still recorded per lane for tail-work accounting but
does not alter the execute mask. Enabled and disabled runs must finish with
bitwise-identical final selected-region labels.

The portable and device records retain one-based convergence rounds, executed
rounds, and tail rounds independently by lane. A zero convergence round means
padding or no proven no-change round before a nonconverged stop. Normal
convergence requires the selected-region slot-equality mask to cover every
valid lane. Maximum-round exhaustion preserves the explicit final slot and
partial labels but does not claim convergence or a bounding-region miss.

### Controls and target classification

Per-round and chunked host control use the same ordered
`(round, controller-advance)` pair as standalone Jacobi. Chunked control queues
the complete configured `K`; pairs after device `done` are no-ops and never
change actual-round parity. Persistent control uses one cooperative kernel.
Every workgroup completes initialization, round work, controller ownership, and
the barriers on both sides of each transition, including the terminal round.
No convergence copy or host synchronization occurs inside that persistent
loop.

Only normal convergence permits target classification. Each lane checks its
own offset target slice against its final slot: all finite means reached, while
any infinity means bounded-region miss. Padded lanes have no result bit.
Maximum-round, invalid-controller, and device-failure exits publish neither
reached nor miss. Target reachability never terminates the fixed-point scan.

### Work records and evidence vocabulary

The portable record keeps shared physical work separate from logical query
work. It reports considered/visited/skipped CSC runs, active lanes summed over
visited runs, shared CSC edge-record loads, logical lane-edge relaxations,
destination/lane writes and decreases, active/valid/padded lane-round capacity,
tail work executed or avoided, exact early-convergence edge work avoided,
union/selected tile-lane positions, modeled selected two-slot device reset
bytes, and source-seed writes. Fresh portable vectors may initialize
nonsemantic host-container cells as a convenience; those writes are not counted
as device reset traffic. All accumulations and byte products are checked.

The HIP metrics retain the corresponding device counters only when Light or
Debug instrumentation requests them. Requested edge/source-distance bytes are
an algorithmic model and remain distinct from profiler-observed cache traffic.
Hardware counters carry an explicit unavailable state until imported from a
separate profiler run. CPU timings and counters are never converted into a GPU
latency, bandwidth, cache, occupancy, or throughput conclusion.

### HIP wave mapping and device-materialized workspace

`BatchedJacobiPullEngine` has a batch-specific interface instead of pretending
that one `BatchDeviceDescription` is one `RouteQuery`. Its dedicated reusable
workspace owns geometrically retained device buffers for terminal payloads and
offsets, dense tile masks, a device-only dense CSC run-mask image,
selected ranges and packed range offsets, the common
controller/status/instrumentation records,
per-lane convergence rounds, batch statistics, and two distance slots. One
lease binds preparation, execution, result transfer, and retirement to one
stream; recovery and teardown fence conservatively only when work may remain.
Selected ranges already carry the union identity and lane masks, so no separate
union-tile device buffer or H2D request exists. The initializer overwrites the
complete controller, and the finalizer overwrites the complete status; Jacobi
preparation therefore does not upload an initial controller or clear status.
Full/evidence execution with `N` ordinary polls performs `N + 1` controller
D2Hs and `N + 1` host synchronizations, with terminal transfer completion and
lease retirement sharing one boundary.

The kernel maps one wave32 to one destination and the first `lane_width` wave
lanes to query lanes. A batch launch rejects a non-wave32 target or an
incompatible block size. The default edge-load strategy leaves the immutable
run mask, source, and weight as compiler-visible uniform loads. A separate
explicit-wave-broadcast variant loads them in lane zero and uses `__shfl`.
Both are retained solely to enable a future controlled target-device
comparison. The broadcast variant is not selected and is not described as an
optimization.

Ordinary and persistent launch legality is derived from the selected real
batch kernel and its shared-memory/block configuration rather than from the
Phase 6 probe. Device-event fields separate preparation, initialization through
terminal status, and result transfer; end-to-end wall time remains separate.
None of those fields has local measured evidence.

### Phase 14 CPU validation boundary

The bounded `bfnew.batched_jacobi_pull` CTest exercises the
width/control/convergence matrix,
independent and multi-source seeding, mixed convergence durations, unreachable
targets, padding, retained and compact run images, exact run/reuse/tail/reset
counters, early-lane parity across later slot swaps, maximum-round behavior,
width-one agreement with standalone Jacobi, and per-lane bounded Dijkstra
comparison. The Release `build/phase14-cpu` suite passed all `15/15` bounded
CTests in `1.38` seconds. Its focused Phase 13+14 matrix passed `3/3` in `0.02`
seconds; `bfnew.batched_jacobi_pull` passed `1/1` in `0.01` seconds in the final
full run. The ASan+UBSan `build/phase14-asan` suite passed all `15/15` in
`2.73` seconds, including the Phase 14 CTest `1/1` in `0.20` seconds.

No HIP compiler, GPU allocation, device correctness run, real query corpus,
uniform-versus-broadcast measurement, register/occupancy sample, profiler
counter, latency, or throughput result belongs to local Phase 14 acceptance.
Those steps remain deferred in `docs/PHASE14_GPU_VALIDATION.md`.

## Implemented Phase 15 overlapping batched dense chaotic push

### One-slot vertex-major atomic lane state

Phase 15 adds `run_host_batched_dense_chaotic_push` for widths 1, 8, 16,
and 32. Width one is a singleton correctness/baseline path; Phase 13's standard
planner family remains 32/16/8. One unsigned IEEE-754 distance word is stored
at `vertex * lane_width + lane`. Every valid lane retains its own flattened
source and target slices, selected tiles, expansion generation, and distance
words. Source seeding writes canonical positive zero only to that lane. Padded
lanes have empty slices and no tile, run, status, or relaxation bits.

The atomic domain is unchanged from standalone dense push: finite nonnegative
values, positive zero, and positive infinity. A lane never reads or updates
another lane's word. There is one in-place slot, no predecessor, no second
distance buffer, no frontier or queue, and no target-triggered early stop.
Width one uses the same CSR schedule and atomic-bit decisions as standalone
`run_host_dense_chaotic_push` and is checked bitwise against it.

### Endpoint-exact CSR scans and convergence

Each selected source row intersects its source-tile lane mask with the current
execute mask. For each outgoing CSR destination-tile run, the already prepared
endpoint mask is intersected once more. A zero mask skips the complete run. A
nonzero mask is reused for all edges in the run; each admitted query lane
atomically loads its independent source word, adds the shared edge weight, and
performs a strict atomic minimum on its independent destination word. The
portable path accepts both retained masks and compact nonzero descriptors and
checks their semantic and logical-work parity. The first HIP path deliberately
accepts only the provisional retained CSR image.

One round is a complete admitted-edge scan. With
`enable_per_lane_convergence=1`, a lane is removed only after its complete scan
publishes no strict decrease. With the flag disabled, valid lanes remain
eligible until a batch-wide no-change scan. The first no-change scan is still
recorded per lane so tail work is auditable. Enabled and disabled execution
must finish with identical selected-region words. A maximum-round stop records
no fabricated no-change scan, no reached bit, and no bounding-region miss.

### Controls, target classification, and work vocabulary

Persistent, per-round-host-poll, and chunked-host-poll modes use the shared
dense controller. Ordinary execution orders one scan and one controller
advance per pair. Chunked control submits complete K-pair chunks for
`K = 2,4,8,16,32`; pairs queued after `done` are byte-preserving no-ops.
Persistent execution is represented as one device-controlled cooperative
launch with barriers around every scan/advance transition. Only normal
convergence permits the final per-lane target slices to be classified as all
reached or bounded-region miss.

`BatchedDenseWorkStatistics` records considered/visited/skipped CSR runs,
active lanes across visited runs, algorithmic CSR edge-record requests,
logical lane-edge relaxations, atomic-compatible source loads, destination
atomic-min attempts, useful updates, active source/lane evaluations, changed
publications, complete scans, convergence tails, and avoided lane-edge work.
The field `csr_edge_loads` is an algorithmic shared edge-record request count.
It is not a measured physical cache load, L2 transaction, or broadcast-
efficiency result. `high_contention_destinations` remains a standalone dense-
engine diagnostic and is not computed by batched Phase 15; zero in that shared
result field means unavailable, not measured zero contention.

Exact integer denominators distinguish valid-lane-round capacity, configured
`lane_width` capacity, full wave32 capacity, unused wave lanes, padded query
lanes, and per-request edge-wave lane capacity. These are utilization
accounting terms, not measured occupancy. Union/selected tile-lane positions
retain the Phase 13 inflation terms. Reset bytes model one selected 32-bit word
per admitted vertex/lane, with source seeds as the zero-valued subset. Fresh
portable-vector initialization outside selected semantic cells is not reported
as device reset traffic. None/Light/Debug instrumentation is tested to preserve
distance bits while exposing its intended counter subsets.

### HIP batch engine and retained workspace boundary

The separately gated `hip::BatchedDenseChaoticPushEngine` consumes one
validated `BatchDeviceDescription` through `run_status_only` or
`run_with_distances`. Its `ReusableBatchedDenseWorkspace` retains flattened
terminals and offsets, union/range metadata and packed offsets, tile masks,
retained CSR masks, controller/status records, optional common and batch
statistics, convergence rounds, and one full graph-sized vertex-major distance
slot. Preparation, execution, result transfer, and retirement share one stream
lease. Initialization writes only selected range/lane words.

The HIP source maps one wave32 to a selected source row and query identities
to wave lanes; the wave services that row's outgoing CSR edge requests.
`BatchedDenseLoadStrategy::compiler_uniform` is the default and leaves
immutable run/edge fields compiler-visible as uniform loads.
`explicit_wave_broadcast` is a separately selectable experiment using wave
broadcast. Both ordinary and persistent variants exist so a later comparison
can hold width, control, convergence, instrumentation, plan, and workload
constant. Actual kernel register count, legal residency, occupancy, L2
reads/writes, atomic/write stalls, and timing require a real HIP compiler and
target device; source structure and CPU counters cannot supply them.

The provisional storage choice remains full-vertex/retained-CSR. Portable
descriptor parity does not prove that compact device storage is faster or even
implemented by the HIP engine. `batched_dense_distance_scratch_bytes` checks
exactly `vertex_count * lane_width * sizeof(uint32_t)`; allocated capacity is
not selected reset traffic and is not a runtime free-memory observation.

### Phase 15 CPU validation boundary

The bounded `bfnew.batched_dense_chaotic_push` CTest exercises widths
1/8/16/32, retained and descriptor CSR images, independent and multi-source
lanes, padding, no-first-decrease/short/several-scan/unreachable lanes,
cross-lane isolation, enabled and disabled convergence, persistent/per-round/
chunked controls including K=2/4/8/16/32, exact logical counters and capacity
identities, instrumentation separation, maximum-round behavior, width-one
standalone parity, and independent bounded Dijkstra comparison.

Final Release evidence: `build/phase15-cpu` passed `16/16` bounded CTests in
`0.17` seconds. The focused `bfnew.batched_dense_chaotic_push` CTest passed
`1/1` in `0.01` seconds. Final ASan+UBSan evidence:
`build/phase15-asan` passed `16/16` in `1.82` seconds, including the Phase 15
CTest `1/1` in `0.14` seconds.

No local Phase 15 command enables HIP or processes the absent real query
artifact. HIP compilation, GPU allocation/execution, device correctness,
compiler-uniform/broadcast comparison, register pressure, actual occupancy,
L2 traffic, write stalls, latency, throughput, and batching benefit are all
deferred and unavailable. The combined-campaign procedure is
`docs/PHASE15_GPU_VALIDATION.md`.

## Implemented Phase 16 overlapping batched active-frontier push

### Lane-private labels and vertex-shared worklists

Phase 16 adds a separate batched frontier path for widths 1, 8, 16, and 32.
The Phase 13 standard planner remains 32/16/8; width one is an explicit
standalone-equivalence baseline. One unsigned IEEE-754 distance word is stored
at `vertex * lane_width + lane`. Flattened source and target arrays retain
their per-lane offsets, and every valid lane seeds all of its canonical sources
to positive zero independently. The same graph vertex may be a source in
several lanes without merging their distance state. Before launch, the HIP
boundary rejects unsorted or duplicate terminal slices, duplicate valid-lane
query IDs, noncanonical padding metadata, and terminals whose owner tile does
not admit that lane.

The hot frontier state has two bounded 32-bit vertex queues and two
graph-sized `LaneMask` arrays. A queue entry identifies one vertex active for
one or more lanes. Its corresponding mask carries those lane identities. The
default queue capacity is one entry per graph vertex, so correct mask
deduplication cannot overflow it; a smaller explicit capacity exists only to
exercise the terminal overflow path. The zero-to-nonzero transition of a next
mask owns the destination's one queue claim. Later successful updates to that
destination can add lane bits and improve independent labels without creating
a duplicate queue entry. No enqueue-generation array is required because the
two mask images follow the same read/write parity as the queues.

Initialization touches selected semantic state only. It writes admitted
vertex/lane distance words, clears both activity masks for every union vertex,
forms the source-lane mask for each distinct source vertex, and appends that
vertex once. Queue storage itself is governed by the controller sizes and need
not be cleared. The same grid-stride initialization also resets every
configured lane's convergence record, including padded lanes. Padded and
nonselected device distance cells are stale but
nonsemantic. The portable result and full downloaded correctness result expose
a complete vertex-by-width image normalized to positive infinity outside every
lane's selected region, then overlay only selected semantic words. This public-
output hardening prevents stale capacity from escaping without presenting the
host normalization as device reset traffic. This dedicated scratch is larger
than Phase 13's generic one-slot label estimate: in addition to
`4 * V * lane_width` distance bytes it includes
`8 * queue_capacity` queue bytes and `8 * V` activity-mask bytes. The Phase 13
estimate explicitly excluded queues and cannot be presented as a complete
Phase 16 capacity result.

### One-thread frontier scheduling and mask deduplication

The round scheduler deliberately retains Phase 11's simplest policy: one
grid-stride thread owns one current worklist entry. That owner reads the
vertex's nonzero current-lane mask and clears the consumed mask so its slot can
be recycled. For each outgoing CSR destination-tile run it computes the
intersection of the current mask, the controller execute mask, and the
prepared endpoint-exact run mask once. A zero result skips the run; the same
nonzero mask is reused for every edge record in the run.

Each admitted lane performs an atomic-compatible load of its source word,
ordinary float addition, and a strict unsigned-word atomic minimum on its own
destination word. Successful lanes for one edge form a bit mask. A nonzero
mask is atomically ORed into the destination's next activity mask; only the
operation observing an old value of zero reserves and writes a queue slot. A
claim at or beyond capacity sets `queue_overflow` and never writes out of
bounds. The round kernel completes all distance, mask, queue, and error writes
before the controller transition consumes them.

The required queue/mask invariant is exact at every round boundary: each
queued vertex is in range and unique, its mask is nonzero and admitted, every
nonzero mask has one queue entry, and the OR of the current masks equals the
controller's execute mask. Queue entries count vertices, not vertex/lane pairs.
The portable engine validates the full invariant set at every bounded round
boundary; its result exposes the final queue and mask images plus per-round
current-size and lane-union traces. The current HIP run output and deferred HIP
CTest expose aggregate/controller guards but not a complete per-round
queue-to-mask boundary ledger. Full device proof therefore requires a future or
manual Debug diagnostic extension during the combined campaign. The design
does not implement prefix-sum edge balancing, virtual warps, a high-degree
path, wave-aggregated queue insertion, or adaptive frontier scheduling. Those
remain future experiments. It also contains no dense-chaotic fallback or
adaptive engine selection.

### Controller convergence and result classification

The batched path has its own frontier controller transition rather than using
the one-lane Phase 11 transition. One executed round increments the count,
swaps the queue and mask slots, clears the recycled queue size and next-round
mask, and selects explicit error, empty-frontier convergence, exact maximum-
round exhaustion, or continuation. Empty-frontier convergence wins if the
last permitted round also empties the queue. Pairs queued after terminal
`done` are byte-preserving no-ops.

With per-lane convergence enabled, a lane absent from the complete next
frontier is removed from the active mask, added to the converged mask, and
never performs later work. With the option disabled, an absent lane still has
no per-vertex mask and therefore receives no invented work; its controller-
level convergence bit is simply deferred until the whole batch queue is empty.
The execute mask always follows actual frontier presence. Consequently both
settings have identical distance and queue semantics. Under this required
policy, work avoided solely by toggling per-lane convergence may honestly be
zero even though first empty-frontier and later batch-tail rounds remain
observable per lane. Per-lane convergence records are written only after a
clean accepted continue, convergence, or maximum-round transition. Error,
invalid-controller, and terminal no-op transitions never publish convergence
proof; a clean maximum-round transition can retain proof for only the lanes
whose complete next frontier is empty.

Per-round host polling submits one ordered `(frontier round, controller
advance)` pair before observing the controller. Chunked polling submits a
complete `K`-pair chunk for `K = 2,4,8,16,32`; later pairs no-op after an early
terminal transition. Persistent control uses one cooperative kernel with
uniform grid barriers between initialization phases, round work, controller
ownership, and terminal status. It performs no host frontier-size or
convergence polling. Final queue parity always comes from the controller. The
ordinary path exposes its actual occupancy- and queue-work-capped launch count
as `ordinary_grid_blocks`; this is launch metadata, not measured occupancy.

Only normal whole-batch convergence permits target classification. Each valid
lane checks only its own target slice: all finite labels set reached, while any
infinite label sets bounded-region miss. Maximum-round, overflow, invalid-
controller, and device-failure stops publish neither. An early-finished lane's
distance words remain frozen while other lanes continue.

### Accounting, retained HIP workspace, and validation boundary

The batch work vocabulary separates shared physical scheduling from logical
lane work. It records frontier vertex entries, active vertex/lane pairs,
multi-lane vertices, considered/visited/skipped CSR runs, active lanes across
visited runs, shared edge-record requests, lane-edge relaxations, successful
lane updates, mask atomics, unique lane enqueue transitions, queue claims,
queue entries saved by lane merging, duplicate suppressions, queue high-water
and overflow, lane utilization/tails, and selected reset/source-seed traffic.
Logical edge requests, atomic attempts, and modeled bytes are not measured L2
transactions, physical memory traffic, occupancy, or timing. Optional hardware
counters remain explicitly unavailable until a matching profiler run exists.
`None` suppresses device algorithm counters. `Light` exposes aggregate work,
sharing, and queue-high-water evidence only. `Debug` adds mask atomics, unique
vertex/lane activations, physical queue claims, lane-merging savings,
same-lane and total duplicate suppressions, and overflow events.

The HIP-gated `BatchedFrontierPushEngine` uses a dedicated
`ReusableBatchedFrontierWorkspace` because the standalone and earlier batch
owners do not contain the queue/mask state or frontier-specific lease. It
retains independent terminal offsets, selected
ranges, tile masks, exact retained CSR masks, controller/status and optional
statistics, per-lane convergence rounds, and the checked frontier scratch.
Preparation validates canonical terminals, query identities, and padding;
execution, result transfer, and retirement remain bound to one stream. The
first device path accepts only the provisional Phase 13 full-
vertex/retained-CSR representation. Portable retained/descriptor equivalence
does not implement compact device storage or establish a production choice.

The bounded CPU acceptance matrix covers widths 1/8/16/32, both convergence
settings, persistent/per-round/chunked controls and all five K values, exact
retained/descriptor parity across the whole configured `V * W` output image
and terminal lane result, independent and multi-source lanes, shared-source
initialization, shared-destination queue merging, padding and nonselected
positive-infinity normalization, mixed frontier durations, unreachable
targets, repeated improvements, high fan-in, zero-weight cycles, run
admission/reuse, portable
queue/mask invariants, initialization and round overflow, instrumentation
separation, clean maximum-round versus error convergence evidence, cross-lane
isolation, width-one standalone parity, and independent bounded Dijkstra
comparison. Every nonselected valid-lane and padded-lane output word is
positive infinity in both retained and descriptor runs.

Final post-hardening HIP-off evidence is Release `17/17` in `1.56` seconds,
including Phase 16 `1/1` in `0.02` seconds, and ASan+UBSan `17/17` in `3.03`
seconds, including Phase 16 `1/1` in `0.16` seconds. The deferred HIP test
passed strict-warnings host syntax and fake-HIP `__HIPCC__` syntax, including
its device probe. Phase 16 public headers passed strict host syntax, and both
production HIP translation units passed strict fake-HIP syntax. These source
checks used no HIP compiler or GPU. No browser, cloud service, or
large/full-graph test was used.

No local Phase 16 command enables HIP or consumes the absent real query
artifact. Device compilation/execution, correctness, real occupancy, physical
memory traffic, latency, throughput, batching benefit, and comparison with the
other batched engines are unavailable and deferred to
`docs/PHASE16_GPU_VALIDATION.md`.

## Implemented Phase 17 batched expansion and all-query execution

### Compact pass boundary and deterministic failed-lane collection

Phase 17 composes the three existing batched engines without changing their
relaxation semantics. `run_batched_expansion` owns canonical query state,
feature construction, overlap planning, batch execution, and retry passes. It
sorts the input by `QueryId`; each pass plans all currently active queries;
each batch returns one compact `DeviceRunStatus`; and only a clean converged
status may partition valid lanes into reached and miss. Reached queries are
accepted immediately. Missed QueryIds from every batch in the pass are
collected, sorted, deduplicated, expanded, and then planned together in the
next pass.

The retry trace records execution ordinal, pass, batch, valid/reached/miss
masks, lane QueryIds and generations, union and selected-region estimates,
work-evidence state, and terminal status. It contains no wall-clock field, so
identical graph/query/configuration inputs have identical traces even if input
query order changes. Outcomes are likewise returned in QueryId order.

Maximum-round and engine errors carry no reach/miss classification and never
enter the expansion path. A callback that publishes classification bits on a
nonclean status, omits a valid lane from a clean reached/miss partition,
returns the wrong engine/control/valid mask, supplies malformed label shape,
or labels unavailable work with nonzero values is rejected at the callback
boundary.

### Schedule geometry, spill admission, and terminal policies

Every `QueryState` freezes the located bounds of the original selected region.
The four schedules advance an intended absolute x/y margin as follows:

```text
one ring:                  1, 2, 3, 4, ...
fixed ring R >= 2:         R, 2R, 3R, 4R, ...
doubling margins:          1, 2, 4, 8, ...
hybrid with S small steps: 1, 2, ... S, 2S, 4S, ...
```

All already admitted tiles remain selected. A retry adds every represented
located tile inside the advanced rectangle. Sparse geometry is explicit: an
intended step that adds no tile is a stalled region, not a successful
expansion. The selected spill tile is preserved. An unselected spill tile has
no invented coordinate and may enter only through actual tile-directory
adjacency from a selected located tile. Long cross-tile edges do not alter the
geometric schedule; they participate when both endpoint tiles are admitted.

The configured maximum counts scheduled geometric expansions. A full-region
fallback is one separate final restart. If its target remains unreachable, the
outcome is `unreachable_in_full_region`. The explicit alternative reports
`expansion_limit` or `region_stalled` without fallback. A query already on the
full tile set is never restarted again. No exit-edge certificate is computed,
so a reached query is accepted on its current induced region and is never
expanded.

### Restart identity and overflow containment

An installed expansion changes only selected tiles and expansion generation.
QueryId, terminal arrays, canonical source/target sets, and terminal maps remain
attached to the query. Before replanning, the prior terminal distance image is
discarded. The portable and HIP adapters rebuild the batch description and
invoke the selected Phase 14–16 engine's selected-only initialization and
original-source seeding, so no label, frontier, changed flag, controller, or
convergence state crosses generations.

The valid-lane mask distinguishes a valid `UINT32_MAX` QueryId from padded
sentinels. If an individual query cannot advance its generation, retry/
expansion count, or margin, it receives `identity_or_count_overflow`, remains
at its prior generation, and does not abort unrelated queries or silently
wrap. Aggregate telemetry, histogram, and campaign counters use checked
accumulation too, but their overflow is not a query disposition: it throws and
aborts the campaign fail-closed.

### Evidence records and schedule selection

`BatchedExpansionMetrics` retains the initial success numerator and input
denominator; final disposition counts; expansion-count histogram; failed-
origin and retry lane utilization; scheduled/fallback totals; repeated
selected-edge estimate; attempted and final tile/vertex/edge work; aggregate
device counters; initial planning, replanning, execution, total host time, and
integer milliqueries/second; plus separately measured retry and failed-batch
shared/logical edge work. Shared work from a mixed failed batch is not falsely
apportioned among lanes.

Every run also computes a deterministic comparison fingerprint from the graph,
initial query identities/regions, engine/control/tuning, planner, limit,
terminal policy, and a required nonzero caller-owned execution-configuration
fingerprint while excluding the candidate schedule. The latter binds runner
choices outside the shared options, such as transfer/load strategy or portable
representation, so unlike executions cannot compare silently. A schedule
record is selectable only when the four schedule kinds have the same nonzero
comparison fingerprint, a nonzero query count, and a valid campaign. Correct
completion ranks first; comparable measured logical/shared work then
participates; deterministic region/retry work handles unavailable-work cases.
If the best complete evidence score is tied, the selector returns no schedule.
No source-level schedule default exists.

### Portable and HIP adapter boundary

`run_host_batched_expansion` prepares retained masks or compact descriptors for
each retry batch and dispatches the chosen portable Jacobi, dense, or frontier
engine. Its optional distance image is projected back into one graph-sized
column per terminal query for bounded Dijkstra comparison.

HIP-gated `BatchedExpansionExecutor` is sequential and non-owning. One instance
binds exactly one resident graph, stream, engine workspace, engine kind, and
run options. Compact mode requires `InstrumentationLevel::none` and calls the
underlying engine's new compact-status path, copying exactly one 48-byte
`DeviceRunStatus` after final classification. It downloads no counter record,
controller, per-lane convergence trace, or distance image. Evidence mode uses
the existing richer status-only paths and reports shared/logical work only when
the underlying path measured it. No new device relaxation kernel or expansion
workspace is introduced.

The deferred `bfnew_gpu_batched_expansion` driver is built only when HIP and
FPGAIF are both enabled. It reads the versioned `.bfgraph` and `.bfqueries`
artifacts, reconstructs and deeply validates the spatial graph, tile runs,
device layout, and every mapped query, fingerprints both files and the device
layout, uploads the graph once, binds the explicitly selected engine
workspace, and runs the complete query artifact through the expansion
controller. Engine and schedule are required command-line choices; the CLI
does not supply a hidden schedule default.

Its `bfnew.batched-expansion.v1` TSV is a fail-closed evidence artifact. The
driver refuses an existing output or stale temporary file, validates canonical
outcome and trace ledgers, writes configuration, fingerprints, metrics,
histogram, query outcomes, and batch traces to a temporary path, and renames
only after a complete write. Engine or per-query identity/count failures
produce a diagnostic report and nonzero exit. Aggregate telemetry overflow
aborts the campaign and cannot be converted into a per-query outcome. Other
terminal dispositions remain explicit report data. The driver's execution-
configuration fingerprint binds its transfer/load choices and is folded into
the schedule-comparison fingerprint. The driver supports compact/None and
evidence/Light-or-Debug modes, but it does not download labels, perform
Dijkstra replay, aggregate statistical repetitions, separate transfer/kernel
timing, or invoke the schedule selector across report files. Those remain
distinct combined-campaign steps. It synchronizes the one resident upload
before entering the controller, then starts the selected reusable workspace
cold and reuses it only within that campaign. Its controller timing therefore
excludes graph upload but includes first workspace growth. The TSV records
`resident_graph_upload=completed-before-controller`,
`workspace_initial_state=cold-reused-within-campaign`, and
`timing_boundary=controller-including-first-workspace-growth`. Those values are
correctness/diagnostic cold-controller evidence, not a warm final-performance
boundary; that requires a later in-process repeated pass with pre-grown state.

The bounded HIP-off matrix exercises all four schedules across all three real
portable engines, multiple and single misses, dissimilar regions, multi-source
restart, long and spill edges, several retries, fallback/unreachable and
explicit failure, deterministic input reorder, retained/descriptor parity,
generation and count boundaries, malformed statuses/evidence, metrics and
fingerprints, invalid-campaign and exact evidence-score-tie rejection, and
exact Dijkstra agreement on final induced regions. Deferred HIP source/runtime
coverage is defined in `docs/PHASE17_GPU_VALIDATION.md`.
The absent real query artifact prevents representative all-query evidence and
therefore prevents a schedule recommendation.

Final HIP-off evidence is Release `18/18` in `1.70` seconds, including Phase
17 `1/1` in `0.01` seconds, and ASan+UBSan `18/18` in `3.32` seconds,
including Phase 17 `1/1` in `0.19` seconds. Strict host and fake-HIP source
checks passed without invoking a HIP compiler or GPU.

## Implemented Phase 18 compact target/path results

### Fixed summary and generation-bound payload

Phase 18 separates compact result identity from the variable path arenas.
`CompactTargetSummary` is a 28-byte device-copyable record containing target,
selected source, target distance, edge-count path length, reached status,
reconstruction status, and an explicit selected-source-valid word. The word is
zero or one; a maximum-valued `VertexId` remains an ordinary strong ID rather
than becoming a sentinel.

`CompactTargetPath` carries only the global reordered path vertices, the
actual final distance label at each of those vertices, and stable logical edge
IDs. Complete shapes are therefore:

```text
vertices.size()        = path_length + 1
distance_labels.size() = path_length + 1
edge_ids.size()        = path_length
```

The aligned labels make exact tightness independently checkable from compact
production output. They do not create a second relaxation column or a
predecessor field. `CompactPathPayload` binds canonical target order and the
unchanged original-terminal map to QueryId and expansion generation while the
engine's final label slot is still alive.

Phase 17's callback result has an optional vector of these payloads and checked
transfer accounting. When compact paths are enabled, a clean callback returns
one payload for every valid lane and none for padding. The orchestrator
validates identities and generations before classifying it. Reached lanes must
be wholly complete. Retry misses discard that generation's complete/
unreachable summaries with its labels, but only after classification proves
that at least one target is unreachable and all others are complete or
unreachable. An all-complete miss or a miss with any reconstruction failure is
normalized to engine failure rather than expanded. A terminal full-region miss
retains its classified payload. This closes the workspace-reuse race:
reconstruction data is captured before another batch can overwrite the engine
scratch. A reached status paired with an incomplete payload is likewise
normalized to engine failure.

### Independent tight-edge backtracking

The portable reconstructor is a new post-relaxation traversal over incoming
CSC. It builds the selected-tile membership mask, verifies the final label
shape, and handles targets in canonical query order. For a finite target it
maintains one backward path and a stack of candidate frames. Incoming edges are
filtered to selected predecessor tiles, tested with exact ordinary-float
addition, and ordered by stable `EdgeId` independently of the CSC's locality
ordering.

A candidate whose predecessor is already on the current path is skipped. If a
chosen predecessor has no remaining route to any canonical source, its frame
is removed and the prior frame tries the next stable edge. This is real
backtracking rather than a greedy predecessor choice, so a zero-weight cycle
or lower-ID tight dead end cannot trap reconstruction. Reversing the successful
backward chain produces source-to-target vertices, compact labels, and edges.
The chosen source is the first vertex. A target already in the source set
produces a zero-edge path.

Every path candidate is selected-region constrained. Unselected labels are
not relied upon as an admission filter: owner-tile membership is checked
explicitly. Global routes outside the admitted tiles therefore cannot enter a
compact path even when they are cheaper.

### Validation and transfer accounting

The ordinary compact validator reconstructs an edge-by-ID endpoint/weight
table from immutable CSR and checks:

- QueryId, generation, disposition, terminal map, and canonical target order;
- known reached/reconstruction enums and a Boolean selected-source word;
- a selected canonical source and finite distance for every complete target;
- path shapes, vertex ranges, simple termination, and selected-tile ownership;
- source and target endpoints plus stable-edge continuity;
- exact tightness of every edge against adjacent compact labels; and
- ordinary-float accumulated edge cost equal to the reported target distance.

The diagnostic validator additionally compares every compact label with the
corresponding word in a supplied full terminal image. That API exists for
bounded correctness replay only. The production validator needs no `V * W`
download.

Checked compact-transfer accounting sums fixed summaries, checked-u32 vertex
IDs, per-path float labels, checked-u32 stable edge IDs, and their total.
Device edge IDs widen to the host's strong 64-bit `EdgeId` only after transfer,
so each path edge occupies four serialized device-payload bytes. Host-resident
query metadata and terminal maps are excluded. This host calculation is a
payload-size model until HIP transfer evidence binds it to physical D2H. That
evidence keeps the compact payload/status/error subtotal separate from any
ordinary or chunked controller-poll bytes and checks the overall D2H sum;
persistent control has no controller polls. The returned compact result owns no
graph-sized distance vector. The portable adapter moves or reconstructs
terminal payloads, creates explicit failure summaries when no valid label image
exists, and releases the diagnostic graph-sized projections after extraction.

### No-congestion assembly and evidence

`run_host_no_congestion_pipeline` is the bounded reference assembly. It forces
generation-bound compact production through Phase 17, consumes each terminal
payload, validates the compact results, and returns no `V * W` label matrix.
It intentionally does not select an engine, control mode, planner, expansion
schedule, or terminal policy; those remain explicit inputs.

`NoCongestionResultAccounting` counts complete, unreachable, terminal-failure,
and reconstruction-failure targets separately. Its portable baseline models
canonical final serialization: one 48-byte `DeviceRunStatus` per declared
batch, each target summary as 28 bytes, each device vertex/stable edge ID as
four bytes, and each aligned label as one float. This model excludes compact
summaries discarded by retries and is not physical D2H evidence. A separate
measured field group accepts cumulative HIP compact payload/status/error
subtotal metrics, controller-poll count/bytes, and overall D2H bytes, including
retries. It checks both the compact subtotal and overall sum; persistent control
must report zero polls, while per-round and chunked controls keep poll traffic
outside the compact subtotal. Unknown statuses, malformed arenas,
contradictory evidence, and count/byte overflow fail closed.

`NoCongestionStageLedger` keeps host-wall and device-event evidence separate.
An unavailable observation cannot carry a numeric value. When planning, SSSP,
geometric region growth, controller/orchestration, reconstruction, and result
transfer all have measured host intervals, `warm_all_query` must be their
checked exact nonoverlapping sum after reusable capacity is already grown. An
asynchronous HIP execution may instead leave host-unpartitionable named stages
unavailable and provide a distinct measured enclosing `warm_all_query`; those
stages remain unavailable rather than becoming measured zero or being folded
into controller/orchestration. `cold_execution` is a separate enclosing
interval for the first all-query execution after upload and may include
workspace construction and capacity growth. `cold_pipeline` adds artifact load
and graph upload to that distinct cold execution; it does not substitute the
warm observation or assert that cold and warm executions are equal. Expansion
contains only measured region-growth work when such a host interval is
available. Individual device events remain individual because summing them is
not a valid aggregate unless a distinct enclosing device observation exists.

The quality helper freezes its sample by sorting SplitMix64 hashes of
`selection_seed ^ QueryId`, independent of input order. It runs unbounded
multi-source Dijkstra for sampled queries, pairs every complete canonical
target (including the complete subset of a terminal mixed full-region result),
and reports absolute inflation plus nearest-rank P50/P95/P99 and maximum cost/
edge-count ratios. Zero over zero is defined as unit relative inflation;
positive bounded cost over a zero unbounded baseline is positive infinity while
its absolute inflation remains finite. These values characterize bounded-
region quality; they never cause a reached query to expand and do not replace
induced-region correctness.

### GPU and campaign boundary

The HIP reconstruction path consumes the exact final distance slot selected by
the engine finalizer. It runs before workspace retirement, ignores padding,
and emits fixed summaries for every clean valid lane followed only by exact
path arenas. Miss summaries support retry/final-unreachable classification;
retry generations are discarded rather than published as terminal results.
Stable incoming edge identity is resident alongside reconstruction CSC data;
device narrowing is checked against the graph's 32-bit-representable image.
Relaxation kernels remain distance-only and unchanged by reconstruction.

The Phase 18 campaign composes artifact load, one graph upload, all planning,
one explicit engine/control configuration, Phase 17 expansion, reconstruction,
and compact result transfer. It records cold artifact load and upload; the
available pre-grown warm component observations; either their required exact
`warm_all_query` sum or a separately measured enclosing warm interval when HIP
host partitioning is unavailable; an independent first capacity-growing
`cold_execution`; and `cold_pipeline` as load plus upload plus that cold
execution. This implementation adds no congestion, resource-conflict state,
exit certificate, early stopping, or adaptive engine switching. The deferred
device and corpus procedure is `docs/PHASE18_GPU_VALIDATION.md`.

The artifact campaign accepts widths 1/8/16/32, with width 1 retained as the
scalar comparison under the same compact-path semantics. It requires measured
GPU-event stage evidence from the cold execution, every additional warmup, and
every measured repetition. A run with unavailable device timing is rejected;
the report never substitutes zero or drops that execution from the evidence
population.

The bounded HIP-off fixture covers stable equal and parallel paths, zero-cycle
backtracking, target-is-source and multi-source identity, duplicate terminal
maps, an excluded cheaper global route, long/spill expansion paths, clean
unreachable and engine failures, payload generation, checked result bytes,
timing-ledger failure modes, deterministic bounded/unbounded quality sampling,
validation corruption, padded lanes, and widths 1/8/16/32 across all three
portable engines. Final evidence is HIP-off Release `19/19` in `2.02` seconds,
including `bfnew.compact_paths` `1/1` in `0.10` seconds; ASan+UBSan `19/19` in
`2.82` seconds, including `bfnew.compact_paths` `1/1` in `0.16` seconds.
Strict host source/public-header syntax and fake-`__HIPCC__` production, CLI,
and deferred-test syntax passed warning-clean.

## Implemented Phase 19 final audit

Phase 19 is an evidence-state layer, not an engine or selector. Its fixed
56-row feature inventory spans the standalone engines and controls,
shootout/profiler infrastructure, overlap planning and workspace models, all
three independently batched engines and widths, expansion/replanning, compact
reconstruction, and the no-congestion campaign boundary. Each feature has
exactly one of four classifications; the local split is 36/0/12/8. Portable
correctness-tested behavior and optional HIP source are separate entries, so a
host pass cannot imply device correctness.

The fixed 27-row comparison inventory includes each engine's best control,
ordinary chunk size, and persistent grid policy; batch width; planner thresholds;
expansion schedule; latency, throughput, cold/warm end-to-end time, miss and
expansion rates, quality, memory, copy/synchronization, and profiler
conclusions. The fixed 10-row profiler inventory is separate. The fixed
question inventory contains the seven Phase 19
cross-engine questions rather than reusing the earlier Phase 12 question IDs.

Unavailable values carry neither numeric payload nor provenance. A measured
comparison, answer, performance-profiled feature, or production recommendation
requires representative device correctness, ordinary timing, and the
applicable trace/PMC evidence under one compatible workload/configuration
identity. A recommendation must pass the existing run, planner, expansion,
mode, and configuration-fingerprint validators. An incomplete candidate
rectangle or absent unique-winner gate selects nothing.

Measured question support references are nonzero, strictly increasing, and
unique within each answer, and the answer fingerprint binds them to the same
evidence snapshot. The snapshot attests a normalized candidate-catalog
fingerprint and complete candidate matrix supplied by a trusted normalized
campaign; this audit does not ingest raw samples, reconstruct catalog
membership, or rerank candidates.

The profiler inventory makes marker-delimited GPU-active time, L2 hit/read/
write behavior, occupancy, memory-unit busy time, waves, and vector/scalar/
memory instruction counts explicit. These schema entries do not become
measurements until a compatible target run is normalized and bound to exact
evidence identities. Measured profiler rows, the profiler-bottleneck
comparison, and the best-engine selection share one configuration fingerprint.

The canonical local snapshot records the actual evidence boundary: the
`.phys` and `.netlist` generation inputs exist, while the versioned
`logicnets_jscl.padding1.v1.bfqueries` artifact, HIP compiler/device run,
representative corpus timing, accepted profiler data, matched CPU baseline,
and normalized Phase 18 provenance/tail attribution do not. It therefore has
zero performance-profiled SSSP features, no measured comparison or answer, no
production configuration, and a deferred hybrid-experiment decision.

Validation requires exhaustive unique IDs in canonical order, exact blockers,
consistent evidence fingerprints/counts, and byte-stable serialization.
Omission, duplication, reorder, unknown enums/records, numeric data on an
unavailable result, duplicate support references, premature measured state,
or a recommendation with any unresolved prerequisite fails closed. The full
matrix and evidence rationale are in `docs/PHASE19_FINAL_AUDIT.md`.

## Phase boundaries

- Work proceeds one accepted phase at a time under
  `GPU_SSSP_CONTINUATION_PROMPT_PACK.md`; the original Phase 6 and later prompts
  are superseded.
- Phase 8 establishes shared resident-layout, API, and workspace foundations.
  Phase 9 implements only standalone Jacobi pull and stops before any dense or
  frontier push work.
- Phase 10 implements only standalone dense chaotic CSR push and stops before
  frontier/worklist state.
- Phase 11 implements only standalone active-frontier/worklist CSR push and
  stops before cross-engine comparison or batching.
- Phase 12 implements only the controlled standalone-engine shootout evidence
  layer and stops before overlap planning, batching, or expansion. Its CPU
  acceptance does not authorize GPU execution before the eventual combined
  campaign.
- Phase 13 implements the deterministic overlapping-batch planner and
  provisional workspace decision. It stops before batched GPU SSSP.
- Phase 14 implements overlapping batched Jacobi pull for widths 1, 8, 16,
  and 32. It stops before the independently implemented batched push engines,
  expansion/replanning execution, or any GPU-performance conclusion.
- Phase 15 implements overlapping batched dense chaotic push for widths 1, 8,
  16, and 32. It stops before batched frontier push, expansion/replanning,
  reconstruction integration, congestion, or any device/performance
  conclusion.
- Phase 16 implements overlapping batched active-frontier push for widths 1,
  8, 16, and 32. It stops before failed-lane expansion/replanning, all-query
  execution, reconstruction integration, congestion, adaptive scheduling, or
  any device/performance conclusion.
- Phase 17 implements reachability-triggered failed-lane expansion,
  deterministic replanning/restart, full-region fallback or explicit failure,
  and all-query execution across the three batched engines. It stops before
  compact path results, reconstruction integration, congestion, or any
  device/performance conclusion.
- Phase 18 implements compact target summaries, post-relaxation path
  reconstruction, generation-bound payloads, exact compact validation, and the
  no-congestion end-to-end result/timing boundary. It stops before congestion,
  resource-conflict logic, and adaptive hybrids.
- Phase 19 audits every implemented and deferred surface, records the final
  insufficient-evidence/no-recommendation result, and stops the campaign. It
  adds no algorithm, congestion state, adaptive hybrid, or Phase 20 work.
- The historical repository may be inspected at the narrow behavioral boundary
  in the project contract but is never a source of copied or transformed code.
- The FPGA24 repository is used only as a read-only external schema/data
  source in the phases that explicitly require it.
