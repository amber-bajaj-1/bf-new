# `bfnew` GPU SSSP Continuation Prompt Pack

## Maintainer execution-policy override — 2026-08-22

For all phases, do not use a browser and do not access `sjc.aupcloud`. Direct
access to the AMD GPU is not available to the coding agent. A phase may be
marked passed when its implementation is complete and the full unchanged CPU
test suite passes. GPU compilation, execution, profiling, and measurements are
optional maintainer-run evidence, not acceptance blockers. When GPU evidence
has not been collected, report that fact explicitly and do not claim that HIP
code or GPU performance was validated.

You are continuing `/Users/amber_bajaj/Desktop/RIPS/bf-new` after the successful completion of Phases 1–5 in `CODEX_IMPLEMENTATION_PROMPTS.md`.

This prompt replaces the original Phase 6 and everything after it. Preserve all accepted Phase 1–5 behavior; do not reimplement those phases.

Use this pack as context, but implement exactly one numbered phase per task. Begin with Phase 6 only. After completing a phase, stop and report its evidence before proceeding.

## Updated global contract

### Scope

The immediate objective is to implement and compare three standalone GPU SSSP engines:

1. synchronous Jacobi min-plus CSC pull;
2. dense in-place chaotic CSR push;
3. active-frontier/worklist CSR push.

Each engine must independently support these control modes:

```text
PersistentCooperative
ChunkedHostPoll(K)
PerRoundHostPoll
```

Do not implement adaptive switching or silently combine engines. A later experiment may compare engines, but each result must identify exactly one engine and one control mode.

After the standalone engines are validated, implement overlapping multi-query SSSP batching separately for all three engines.

Do not implement:

- PathFinder congestion costs;
- occupancy or historical congestion updates;
- routing-resource disjointness;
- disjoint-region batching;
- an adaptive push/pull hybrid;
- negative weights;
- target-based early stopping without a separate proof;
- global-optimality certification outside the selected bounding region.

The future congestion-consistency point will be a batch boundary, but no congestion state or placeholder congestion code should be added now.

### Cost and correctness semantics

- Costs are finite, nonnegative `float` values.
- Costs are frozen for the entire duration of one SSSP query or query batch.
- Use ordinary IEEE-754 float addition and strict `<`, matching Phase 5.
- Preserve multi-source queries. A logical FPGA net can expose one or two valid RRG source vertices; seed every valid source with zero.
- Exclude driverless nets and `GLOBAL_USEDNET` from route queries.
- The hot relaxation state stores distances only. Do not add predecessors to relaxation records.
- Use the Phase 5 Dijkstra oracle and post-relaxation reconstruction for correctness.
- Exact representable tests require bitwise equality. General-data reporting may use the existing ULP-aware comparison, never as an algorithmic decision.

### Bounding-region semantics

A query is solved on the induced subgraph of its selected tiles. An edge is admitted only when both endpoints are admitted for that query.

A result is considered correct when it equals Dijkstra on that same induced subgraph. The selected bounding region is an intentional heuristic approximation of the full routing graph.

Global shortest-path optimality is not required. If all targets are reached within the current region, accept the result even if an unbounded graph contains a cheaper route.

Do not add an exit-edge lower-bound certificate or expand a query merely because a cheaper route might exist outside its selected region. Unbounded Dijkstra may be used on documented samples only to report path-quality inflation; it is not the correctness oracle for bounded execution.

If any target is unreachable:

1. mark that lane as a bounding-box miss;
2. finish the current batch;
3. collect one compact failed-lane/query list;
4. expand all failed queries;
5. deterministically replan them into new overlapping batches;
6. restart those queries from their original source sets.

Do not perform per-round bounding expansion. Do not retain labels across expansion until that has a separate correctness argument and benchmark.

A final full-region fallback may be used only after a configurable measured expansion limit, so every routable query eventually has a completion path.

### Reuse policy

The historical repository:

```text
/Users/amber_bajaj/Desktop/RIPS/rips2026-amd-routing
```

remains read-only. It may be inspected to understand externally observable behavior, documented artifact formats, FPGAIF ingestion requirements, device-graph construction semantics, terminal mapping, and compact path-processing concepts.

No historical source text, tests, implementation fragments, constants, data tables, controller logic, synchronization logic, queue logic, workspace implementation, or GPU kernel code may be copied, pasted, translated, mechanically transformed, or transplanted into `bf-new`.

Behavioral concepts and documented artifact/interface formats may be adapted only through an independent new implementation in `bf-new`, with attribution to the inspected behavior where appropriate. A new adapter must be written against documented artifacts or the FPGA Interchange schemas. The new GPU engines and all control flow must be designed from the Phase 5 semantics and this prompt.

Prefer a narrow adapter over a broad compatibility layer. Every independently implemented adapter must have focused validation proving that it produces the current `WeightedGraph`, `PartitionedGraph`, and query semantics.

The FPGA24 repository remains read-only:

```text
/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest
```

### Known target

The validated AUP environment is:

```text
GPU target:                 gfx1151
Architectural CUs:          40
HIP scheduling units:      20 WGPs
Wave size:                  32
Cooperative launch:        supported
Cooperative grid sync:     successfully executed
Probe occupancy ceiling:   8 blocks/WGP, 160 blocks total
ROCm:                       7.13.0
ROCprofiler-SDK:            1.3.0
PC sampling:                not exposed
```

`hipDeviceProp_t::multiProcessorCount` is 20 because the device operates in WGP mode. Use that value when calculating cooperative grid limits.

The probe’s 160-block ceiling applies only to the trivial probe. Every real kernel must run:

```cpp
hipOccupancyMaxActiveBlocksPerMultiprocessor(...)
```

on its own kernel and launch configuration. Never hard-code 160.

Wave32 is the primary execution width. Use 32 as the principal batching width. Width 64 is a separate two-wave experiment, not the assumed default.

### Common GPU engine interface

Introduce one shared public interface without merging kernel semantics:

```text
EngineKind:
  JacobiPull
  DenseChaoticPush
  FrontierPush

ControlMode:
  PersistentCooperative
  ChunkedHostPoll
  PerRoundHostPoll

GpuRunOptions:
  engine
  control_mode
  rounds_per_chunk
  block_size
  blocks_per_wgp or occupancy-derived grid policy
  instrumentation_level
  maximum_rounds
  selected-region information
  enable_per_lane_convergence (batched engines only; default true)

RouteQuery:
  stable query ID
  one or more source vertices
  one or more target vertices
  selected tiles / bounding region
  expansion generation

GpuSsspResult:
  final distance-buffer identity
  convergence status
  completed rounds
  reached-target mask
  bounding-box miss status
  valid/active/converged lane masks for batched execution
  work statistics
```

Standalone engines may share:

- immutable device graph storage;
- allocation and stream wrappers;
- query validation;
- timing utilities;
- result verification;
- profiling infrastructure;
- post-relaxation path reconstruction.

They must not share an ambiguous “relaxation engine” that changes semantics based on runtime state.

### Device controller and round protocol

Use one explicit device-resident controller representation for all control modes. Standalone execution uses lane bit zero; batched execution uses one bit per valid lane.

The controller must contain at least:

```text
valid_lane_mask
active_lane_mask
changed_lane_mask
converged_lane_mask
execute_lane_mask
rounds_completed
maximum_rounds
distance_read_slot
distance_write_slot
frontier_read_slot
frontier_write_slot
frontier_size[2]
next_frontier_lane_mask
done
stop_reason
error_bits
```

Use explicit stop reasons that distinguish normal convergence, maximum-round exhaustion, queue overflow, invalid controller state, and device failure. Algorithm or controller failures must not be reported as bounding-box misses.

Initialize the controller once before the first round. It must set valid and active lanes, clear convergence/change/error state, select the initial distance and frontier slots, and prepare the first round.

For ordinary noncooperative execution, one logical round is exactly:

```text
engine round kernel
controller advance kernel
```

The engine round kernel:

- reads only the controller's `execute_lane_mask` and current buffer/queue slots;
- cheaply no-ops when `done` is set or `execute_lane_mask` is zero;
- performs one complete engine-specific round for every executing lane;
- accumulates lane-specific changes or next-frontier lane presence on the device;
- never swaps buffers, swaps queues, or makes a host-side convergence decision.

The controller advance kernel:

- consumes the completed round's change or next-frontier state;
- updates active and converged lane masks according to `enable_per_lane_convergence`;
- swaps Jacobi distance slots after every executed Jacobi round;
- swaps frontier slots after every executed frontier round and clears the recycled next queue/mask state;
- leaves dense-chaotic distances in their single in-place slot;
- increments `rounds_completed` only for a round that actually executed;
- clears the change/next-round state and prepares `execute_lane_mask` for the following round;
- sets `done` only for convergence, maximum-round exhaustion, or an explicit error;
- is itself a cheap no-op after `done` is set.

For Jacobi, every executing lane must write its complete admitted `d_next` column from `d_old`, including the unchanged `d_old[v]` candidate. When a lane converges, its two distance-buffer columns must therefore be bitwise identical before that lane is removed from `active_lane_mask`. This invariant allows later batch-wide slot swaps without corrupting already-converged lanes.

For dense chaotic push, a lane converges only when one complete admitted-edge scan produces no strict decrease for that lane.

For frontier push, lane-specific next-frontier presence determines which lanes remain active. When per-lane convergence is disabled, do not synthesize work for a lane with no frontier entries; simply defer controller-level lane completion until the complete batch frontier is empty.

When `enable_per_lane_convergence` is true, a lane that produced no change in a complete Jacobi/dense round, or no next-frontier entry in a frontier round, is removed from `active_lane_mask` and its distance state is frozen. When false, Jacobi and dense execution retain all valid lanes until a complete batch round produces no changes in any lane. Both settings must produce identical final distances.

Persistent cooperative kernels must implement the same state transitions inside the kernel. Use grid barriers around round work and controller advancement, and allow only one designated controller owner to perform global slot/mask transitions. All workgroups must participate in every grid barrier even after `done` becomes true.

### Tile edge-run metadata

Exploit the Phase 3 ordering in which outgoing CSR edges are grouped by destination tile and incoming CSC edges are grouped by source tile. Build immutable maximal tile-run metadata without changing the accepted Phase 1–5 graph semantics.

Provide structure-of-arrays metadata equivalent to:

```text
CSR:
  csr_row_run_offsets[vertex_count + 1]   // indexes CSR runs by source row
  csr_run_edge_offsets[csr_run_count + 1]
  csr_run_destination_tile[csr_run_count]

CSC:
  csc_column_run_offsets[vertex_count + 1] // indexes CSC runs by destination column
  csc_run_edge_offsets[csc_run_count + 1]
  csc_run_source_tile[csc_run_count]
```

The owner tile of the CSR source row or CSC destination column comes from existing vertex metadata and need not be duplicated per run. A run must never cross a row/column boundary even when adjacent rows/columns refer to the same remote tile.

Validate that runs are deterministic, nonempty, maximal within each row/column, cover every sparse-layout edge exactly once in layout order, retain parallel edges, and correctly include long cross-tile and spill-tile edges. Keep 64-bit host offsets. A checked 32-bit device representation may be evaluated under the same narrowing rules as other device offsets.

For a batch, compute the applicable lane mask once per run:

```text
CSR run mask = source-owner-tile lane mask & destination-tile lane mask
CSC run mask = destination-owner-tile lane mask & source-tile lane mask
```

Kernels must skip a complete run when this mask is zero and reuse the nonzero mask for all edges in that run. Do not perform source/destination tile lookup or endpoint-admission mask intersection independently for every edge. The batch representation may materialize compact `(run_id, lane_mask)` descriptors or a reusable per-run mask array; Phase 13 must measure and document that representation choice.

### Three control modes

#### Persistent cooperative

- Launch one cooperative kernel for one query or query batch.
- Perform convergence entirely on the GPU.
- Use `cooperative_groups::this_grid().sync()` for inter-round ordering.
- All workgroups must reach every grid barrier; no workgroup may return early.
- Copy status/results to the host only after the kernel completes.
- Calculate the legal resident grid using the actual kernel and its block size.
- Benchmark multiple legal grid sizes rather than assuming maximum residency is fastest.

#### Chunked host polling

For a configurable `K`:

- enqueue up to `K` ordered `(engine round kernel, controller advance kernel)` pairs in one stream;
- keep convergence state on the device;
- make every later queued pair cheaply no-op when an earlier controller advance set `done`;
- perform one status copy and one host synchronization per chunk;
- do not synchronize between rounds in a chunk.

Test at least:

```text
K = 2, 4, 8, 16, 32
```

The stream order and kernel boundaries provide inter-block ordering between each round and controller advance; they do not require the CPU to synchronize. The one status copy occurs only after the final queued controller advance. The host reads final distance/frontier slot identity from the copied controller and must never infer parity from the requested value of `K`.

#### Per-round host polling

This is the historical-control baseline:

- enqueue one `(engine round kernel, controller advance kernel)` pair;
- copy the minimal convergence/controller state;
- synchronize the stream;
- decide on the host whether to launch another round.

Do not add unrelated CPU work. Count every synchronization, controller copy, and kernel dispatch.

### Instrumentation

Provide instrumentation levels:

```text
None:
  production timing; no expensive device counters

Light:
  rounds, edges examined, successful decreases, active vertices/lanes,
  queue sizes, host checks, dispatches, expansion count

Debug:
  atomic attempts, successful atomic updates, queue claims,
  duplicate suppressions, mask operations, and overflow diagnostics
```

Instrumentation must be toggleable because global atomic statistics can distort performance.

No available ROCprof counter directly measures algorithmic atomic attempts, queue deduplication, or cooperative barrier cost. Those require explicit instrumentation or isolated microbenchmarks.

### ROCprofiler protocol

The `-d` option is global and must precede the subcommand:

```bash
rocprofv3-avail -d 0 list --pmc
rocprofv3-avail -d 0 info --pmc
rocprofv3-avail -d 0 pmc-check <counter names>
```

Before profiling engines, use `pmc-check` to find compatible passes. Begin with these candidate groups and split them until every pass is accepted:

```text
Utilization:
  GRBM_COUNT
  GRBM_GUI_ACTIVE
  GPUBusy
  OccupancyPercent
  MeanOccupancyPerCU
  MeanOccupancyPerActiveCU
  MemUnitBusy

Instruction/wave:
  SQ_WAVES_sum
  SQ_WAVE_CYCLES
  SQ_BUSY_CYCLES
  SQ_INSTS_VALU
  SQ_INSTS_SALU
  SQ_INSTS_SMEM
  SQ_INSTS_FLAT
  SQ_INSTS_TEX_LOAD
  SQ_INSTS_TEX_STORE

L2 and memory traffic:
  FETCH_SIZE
  WRITE_SIZE
  L2CacheHit
  GL2C_HIT_sum
  GL2C_MISS_sum
  GL2C_MC_RDREQ_sum
  GL2C_MC_WRREQ_sum
  GL2C_EA_WRREQ_64B_sum
  GL2C_WRREQ_STALL_max
  WriteUnitStalled
```

Record accepted counter groups in `docs/PROFILER_COUNTERS.md`.

Use separate runs for timing and counter collection. Profiler-instrumented runtimes must not be reported as ordinary performance.

Use tracing to count host dependencies:

```bash
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-format csv \
  -- <benchmark command>
```

### Timing protocol

- Keep the graph resident before timed queries.
- Separate graph construction, graph upload, query preparation, batch planning, SSSP, expansion, reconstruction, and result-transfer times.
- Report both GPU-event time and host wall time.
- Warm up the GPU.
- Interleave engine/control configurations where practical to reduce thermal and clock-order bias.
- Report at least P50, P95, and P99, plus total throughput.
- Report profiler runs separately from unprofiled timing.
- Make no bottleneck claim from one counter alone.

### Process rules

Before editing:

1. read `CODEX_IMPLEMENTATION_PROMPTS.md`;
2. read `PROJECT_CONTRACT.md`;
3. read `docs/DESIGN.md`;
4. read `docs/IMPLEMENTED_OPTIMIZATIONS.md`;
5. inspect the Phase 1–5 interfaces and tests;
6. run the existing CPU suite unchanged.

Implement exactly one phase per task.

Under the maintainer execution-policy override above, a HIP phase may be
accepted from a passing full CPU suite without direct AUP execution. Preserve
optional `gfx1151` test instructions, distinguish unvalidated GPU behavior from
CPU-tested behavior, and never infer GPU performance from CPU results.

Every phase handoff must include:

- files changed;
- commands run;
- CPU and GPU test results;
- correctness comparisons;
- measured findings;
- profiler evidence when required;
- remaining risks;
- decisions deferred;
- exact next phase number.

Do not describe an optimization as implemented without source, passing tests, and—when performance-related—measurements.

---

# Phase 6 — campaign reset, optional HIP build, and profiling gate

Update the project contract and documentation to reflect this replacement campaign:

- replace the former blanket clean-room wording with the inspection-and-independent-adaptation policy in this prompt, while preserving the prohibition on copying or mechanically transforming historical source, tests, fragments, constants, or control logic;
- retain read-only ownership of the historical and FPGA24 repositories;
- allow a narrow new adapter based only on documented artifacts, FPGA Interchange schemas, and independently reimplemented observed behavior;
- replace global exact-region certification with reachability-triggered batched expansion;
- remove adaptive push/pull and disjoint-batching goals;
- record that costs are immutable during queries;
- state explicitly that congestion is deferred;
- record the measured AUP hardware and wave32 behavior.

Add optional HIP support to CMake without breaking CPU-only builds. A CPU-only configuration must continue to build on systems without ROCm. A HIP-enabled configuration must fail clearly when requested but unavailable.

Add a small GPU capability/profiling tool that records:

- target architecture;
- wave size;
- WGP/multiprocessor count;
- architectural CU count when available;
- cooperative-launch flag;
- real cooperative launch result;
- occupancy for the probe;
- profiler version;
- accepted PMC groups.

Run `pmc-check` for the candidate counter groups, splitting incompatible groups. Run one actual counter collection on a nontrivial repeated kernel, not only the one-barrier probe.

Add a cooperative-barrier microbenchmark varying:

- block size: 128, 256, 512 where legal;
- resident blocks per WGP: 1, 2, 4, and maximum where legal;
- number of grid barriers.

Do not implement SSSP in this phase.

Acceptance:

- existing Phase 1–5 CPU tests remain unchanged and pass;
- HIP-enabled build runs on `gfx1151`;
- cooperative capability and actual execution pass;
- accepted PMC groups are recorded;
- at least one PMC collection produces nonempty counter data;
- barrier cost is reported without claiming it predicts SSSP performance.

Stop after Phase 6.

---

# Phase 7 — real FPGA workload bridge and query corpus

Introduce the real workload before optimizing GPU kernels.

First perform a read-only audit of the historical preprocessing pipeline. Report the narrowest practical reuse boundary:

1. consume an existing documented graph/query artifact through an adapter;
2. independently implement a new bridge against the FPGA Interchange schemas while matching validated observable parsing, graph-construction, and terminal-mapping behavior;
3. define a new documented intermediate artifact and adapter only if neither previous option can preserve the Phase 1–5 graph invariants.

Do not copy or mechanically transform historical code or tests at any boundary. Do not reuse historical SSSP, controller, synchronization, queue, workspace, or GPU-kernel implementation. Behavioral observations may inform the new implementation, but the new adapter must compile and validate without any historical source dependency.

If choosing among these options would require a large incompatible rewrite, stop after the audit and ask the user to approve the recommended boundary.

Produce a deterministic bridge to the current `WeightedGraph`, `PartitionedGraph`, and new `RouteQuery` representation.

Required behavior:

- load the reusable xcvu3p routing graph;
- preserve directed PIP orientation and stable physical provenance;
- construct or validate both CSR and CSC;
- map physical-net source forests and stubs to RRG vertices;
- preserve one or two RRG sources per ordinary net;
- exclude `GLOBAL_USEDNET`, static preservation nets, driverless nets, and unsupported partially routed shapes;
- retain all requested sinks;
- retain terminal coordinates and tile ownership;
- emit deterministic versioned graph/query artifacts under `bf-new/out`;
- never modify the input `.phys` or `.device` files.

Create the complete query corpus for `logicnets_jscl`, then validate at least one medium benchmark. Scan all 13 inputs for metadata even if full graph execution is deferred.

Report:

- query count;
- one-source versus two-source distribution;
- sink/fanout distribution;
- initial bounding-tile counts for several padding values;
- selected vertex/edge estimates;
- missing-coordinate rates;
- box-overlap and tile-set Jaccard distributions;
- predicted batch-width occupancy for 8, 16, and 32 lanes;
- potential spill-tile admissions.

Acceptance:

- every emitted query has at least one valid source and target;
- graph and tile-directory deep validators pass;
- repeated artifacts are deterministic;
- sampled mapped queries agree with independently inspected FPGAIF endpoints;
- Phase 5 Dijkstra runs correctly on bounded real-query samples;
- the real workload is available before any production GPU engine is written.

Stop after Phase 7.

---

# Phase 8 — shared resident device graph, query API, and workspace foundation

Implement shared GPU infrastructure without implementing an SSSP engine.

Add:

- HIP error and stream RAII;
- persistent device allocations;
- immutable device CSR and CSC views;
- immutable CSR destination-tile-run and CSC source-tile-run metadata from the global contract;
- query/source/target buffers retained across calls;
- selected-region/tile-mask buffers;
- reusable run-lane-mask or compact active-run descriptor buffers;
- controller/status buffers;
- reusable workspace reservation;
- asynchronous upload/download helpers;
- GPU-event and wall-time measurement;
- optional instrumentation buffers.

Keep the host `WeightedGraph` representation unchanged.

The hot device views should omit fields not used during relaxation. In particular, do not load stable edge IDs or provenance in distance-only kernels.

Evaluate a checked device-only 32-bit offset representation because the current FPGA graph fits within 32-bit edge positions. Preserve 64-bit host offsets and reject conversion when counts do not fit. Do not assume 32-bit is faster without measuring it later.

Define the common engine and control-mode API from the global contract.

Ensure:

- graph upload occurs once;
- no query performs graph allocation or upload;
- workspace growth is amortized and retained;
- different engines may reuse an allocation pool but cannot accidentally read each other’s stale state;
- memory requirements are reported before allocation;
- only the active engine’s large workspace must be resident unless simultaneous residency is justified.

Test byte-for-byte device round trips on:

- Phase 5 fixtures;
- spatially reordered graphs;
- a bounded real graph;
- multiple sources and targets;
- empty rows/columns;
- spill-tile metadata.

Add deep host and device validation of tile runs, including empty rows/columns, internal runs, adjacent-tile runs, long cross-tile runs, parallel edges, and spill-tile runs. Demonstrate that run-level admission produces the same admitted edge set as endpoint-by-endpoint admission on Phase 5 fixtures and bounded real-graph samples.

Acceptance:

- all CPU tests pass with HIP disabled;
- HIP-enabled transfer tests pass on AUP;
- resident graph memory is reported by component;
- tile-run metadata is deterministic, fully validated, and resident with the graph;
- no SSSP kernel exists yet;
- no per-query graph copy or allocation appears in a HIP trace.

Stop after Phase 8.

---

# Phase 9 — standalone GPU-controlled Jacobi CSC-pull engine

Implement only `JacobiPull`.

Semantics:

```text
d_next[v] =
  min(
    d_old[v],
    min over admitted incoming edges (u,v):
      d_old[u] + weight(u,v)
  )
```

Requirements:

- use incoming CSC;
- use separate `d_old` and `d_next` buffers;
- assign one logical owner to each destination;
- do not use atomic distance updates;
- process only destinations admitted to the selected region;
- traverse CSC source-tile runs and skip a complete run when its run admission mask is zero;
- ignore an incoming edge when its source is not admitted, with admission decided once per run rather than once per edge;
- seed all valid sources with zero;
- preserve strict float comparison;
- use per-block reduction before updating the global changed flag;
- converge until no label changes;
- do not add target early stopping.

Implement all three control modes.

Persistent-specific requirements:

- use grid-stride destination traversal;
- reset convergence state on the GPU;
- ensure every workgroup reaches every grid barrier;
- swap old/next roles uniformly after a grid barrier;
- make final-buffer parity explicit in the result;
- compute occupancy from the real kernel;
- test multiple legal cooperative grid sizes.

Chunked ordinary kernels must use the exact engine-round/controller-advance protocol from the global contract without host synchronization inside a chunk. Already-converged rounds must no-op using device controller state, and final-buffer identity must come from the controller.

Tests:

- every Phase 5 fixture;
- long chains;
- disconnected graphs;
- zero-weight cycles;
- parallel edges;
- multiple sources;
- a box that excludes a cheaper global route;
- a box miss;
- random seeded weighted graphs;
- all three control modes;
- `K = 2,4,8,16,32`;
- equality among control modes and with bounded CPU Dijkstra.

Profile:

- ordinary-round dispatch cost;
- cooperative grid-barrier cost;
- examined CSC edges;
- useful decreases;
- L2 hit/miss and read traffic;
- occupancy and waves;
- GPU and wall time.

Acceptance:

- bitwise agreement on representable tests;
- documented ULP comparison on general random tests;
- no host interaction during persistent convergence;
- no distance atomics;
- no single-buffer read/write race;
- all control modes remain independently selectable.

Stop after Phase 9.

---

# Phase 10 — standalone dense chaotic CSR-push engine

Implement only `DenseChaoticPush`.

Each round scans every admitted outgoing edge and performs an in-place atomic decrease:

```text
candidate = atomic-compatible-load(d[u]) + weight(u,v)
atomicMinNonnegativeFloat(d[v], candidate)
```

Requirements:

- use outgoing CSR tile runs or a measured flat edge-run view;
- compute admission once per CSR destination-tile run and skip zero-mask runs;
- scan the entire admitted edge set each round;
- do not use a frontier or worklist;
- do not call this deterministic Gauss–Seidel;
- call it chaotic/asynchronous push;
- convergence occurs after a complete admitted-edge scan produces no decrease;
- use device-resident changed/done state;
- do not store predecessors.

Because all distances are finite nonnegative floats or positive infinity, a 32-bit unsigned representation may use monotonic IEEE float bit ordering. Prove this restriction in tests.

Do not mix ordinary non-atomic reads with atomic writes to the same distance storage. Use an atomic-compatible load with documented HIP semantics.

Implement all three control modes with the same public options as Jacobi.

Add explicit instrumentation for:

- atomic attempts;
- successful decreases;
- high-contention destinations;
- changed-flag updates;
- full-edge rounds.

Tests must include:

- all Jacobi correctness tests;
- a high-fan-in contention graph;
- a hub/high-fan-out graph;
- adversarial scheduling with long chains;
- repeated deterministic runs;
- consistency among persistent, chunked, and per-round control;
- bounded CPU Dijkstra agreement.

Profile:

- write traffic;
- L2 traffic;
- `SQ_INSTS_TEX_LOAD` and `SQ_INSTS_TEX_STORE`;
- occupancy and waves;
- atomic-attempt/success ratio;
- full-edge bandwidth;
- control-mode synchronization cost.

Acceptance:

- eventual final distances match bounded Dijkstra;
- no frontier state exists;
- nondeterministic round progression is allowed, but final results are deterministic;
- atomic memory semantics are documented and stress-tested;
- all control modes remain selectable.

Stop after Phase 10.

---

# Phase 11 — standalone frontier-push engine

Implement only `FrontierPush`.

Semantics:

- the initial frontier contains every canonical source;
- a round relaxes outgoing edges of the current frontier;
- a successful strict atomic decrease activates the destination for the next frontier;
- frontier exhaustion means convergence;
- different sources and targets remain part of the same query, not separate searches.

Keep all queue state on the GPU.

Required state:

- current and next frontier storage;
- current and next sizes;
- distance state;
- deduplication/generation state;
- touched/initialized state if needed;
- explicit overflow detection.

A vertex may appear at most once in the next frontier. A later improvement may still update its distance without adding a duplicate entry.

Implement all three control modes. Persistent mode must swap queues and test exhaustion entirely on the GPU.

Use the simplest correct one-thread-per-frontier-vertex mapping in this campaign. Within each active CSR row, consume destination-tile runs and reuse the run admission decision across the run.

Do not implement or benchmark prefix-sum edge balancing, virtual-warp/high-degree scheduling, wave-level successful-lane aggregation, or adaptive frontier scheduling in the current phases. Record these as explicit future optimization candidates, and keep queue and run interfaces narrow enough that a later campaign can add them without changing distance or frontier semantics.

Instrumentation:

- frontier vertices per round;
- active outgoing edges;
- maximum queue size;
- queue claims;
- duplicate suppressions;
- atomic attempts and successes;
- empty/small-frontier rounds;
- overflow status.

Tests:

- sparse wavefront chain;
- rapidly expanding grid;
- high-fan-in convergence;
- repeated improvements to one queued vertex;
- zero-weight cycles;
- frontier empty at initialization only when inputs are invalid;
- all control modes and K values;
- exact agreement with bounded Dijkstra.

Acceptance:

- no host queue-size polling in persistent mode;
- no silent queue overflow;
- frontier semantics are separate from dense chaotic push;
- all three control modes remain selectable;
- deferred frontier scheduling and wave-aggregation optimizations are documented as not implemented.

Stop after Phase 11.

---

# Phase 12 — controlled single-query engine shootout

Do not add a fourth algorithm or adaptive switching.

Build a reproducible benchmark comparing:

```text
JacobiPull × three control modes
DenseChaoticPush × three control modes
FrontierPush × three control modes
```

Use:

- synthetic sparse/dense-frontier adversarial cases;
- at least 1,000 representative `logicnets_jscl` queries;
- stratification by box vertices, box edges, fanout, source count, and expected rounds;
- identical graph, weights, boxes, sources, and targets;
- interleaved execution order.

Sweep:

- block sizes 128, 256, and 512 where legal;
- persistent resident blocks per WGP;
- `K = 2,4,8,16,32`;
- instrumentation disabled for final timing.

Report:

- P50/P95/P99 single-query wall and GPU time;
- total throughput;
- rounds;
- examined edges;
- useful-decrease ratio;
- frontier work;
- atomic work;
- kernel dispatches;
- host synchronizations;
- controller copies;
- L2 behavior;
- occupancy;
- memory-unit activity;
- instruction mix;
- long-tail queries.

Answer these measured questions:

1. Does persistent cooperative control beat chunked launches?
2. What K minimizes wall time for each ordinary-kernel engine?
3. Does maximum cooperative occupancy help or hurt?
4. Where does dense pull beat sparse frontier push?
5. Is dense chaotic push useful beyond being a diagnostic?
6. Are the expensive tail queries associated with low locality, excessive rounds, atomics, or box size?
7. How much of old per-round polling time is eliminated?

Retain every toggle regardless of which becomes the default.

Acceptance:

- correctness passes before timing;
- profiler and timing runs are separate;
- every conclusion names its workload and counter evidence;
- no hybrid is implemented;
- recommended defaults are recorded but remain configurable.

Stop after Phase 12.

---

# Phase 13 — deterministic overlapping-batch planner and workspace decision

Implement batching plans and workspace descriptions, but no batched GPU SSSP kernel yet.

Every query must be assigned exactly once per routing pass, including singleton leftovers.

Do not merge sources between queries. Each lane has independent distance state.

Use actual selected tile sets and admitted-edge estimates, not only terminal rectangles.

Planner inputs:

- query ID;
- source/target tiles;
- selected tiles;
- selected vertex and edge estimates;
- tile-set Jaccard overlap;
- union-tile inflation;
- source count;
- target count;
- expansion generation.

Planner behavior:

- prefer 32-lane batches on wave32;
- also produce 8- and 16-lane plans;
- allow singleton and padded final batches;
- defer 64 lanes to a two-wave experiment;
- group by overlap and bounded union inflation;
- be deterministic under ties;
- never omit a query.

Device batch representation:

```text
query IDs by lane
sources/targets with per-lane offsets
union tile list
per-tile active-lane mask
active CSR/CSC run descriptors or reusable per-run lane masks
selected vertex ranges
selected edge estimates
lane validity mask
reached/miss mask
```

Evaluate two workspace strategies before choosing:

1. full persistent graph-sized vertex-major lane arrays with selected-tile reset;
2. compact union-tile arrays with global/local mapping.

Measure:

- allocation size;
- reset traffic;
- mapping/build time;
- selected versus wasted vertices;
- ability to reuse allocations;
- 8/16/32-lane memory;
- maximum concurrent workspace count.

Prefer the simpler full layout if capacity is safe and reset/scanning costs are acceptable. Prefer compact union storage only if measured savings justify mapping complexity.

For each workspace strategy, also evaluate whether materialized compact `(run_id, lane_mask)` descriptors or a retained per-run mask array gives the simpler and faster measured execution preparation. Whichever representation is selected must allow kernels to skip zero-mask runs without repeating tile admission work per edge.

Acceptance:

- deterministic golden planner tests;
- all real queries assigned exactly once;
- one- and two-source queries retained correctly;
- lane masks agree with query tile sets;
- run masks agree exactly with endpoint-based admission and cover all admitted edges;
- memory estimates validated before allocation;
- a documented workspace decision based on actual measurements.

Stop after Phase 13.

---

# Phase 14 — overlapping batched Jacobi pull

Implement batched `JacobiPull` for widths:

```text
1, 8, 16, 32
```

Use vertex-major distances with contiguous lanes and two distance buffers.

A destination’s admitted lane mask is determined by its tile. An incoming edge applies only to lanes admitted on both source and destination tiles.

Traverse incoming CSC source-tile runs. Intersect endpoint-tile lane masks once per run, skip zero-mask runs, and reuse the resulting mask for every edge in the run.

Map wave32 lanes to query lanes so that incoming edge/source/weight information can be shared or broadcast where profitable. Measure explicit shuffle/broadcast against compiler-generated uniform loads before retaining it.

Requirements:

- lane sources are seeded independently;
- no sources are merged;
- inactive/padded lanes perform no semantic work;
- support `enable_per_lane_convergence=true|false`, defaulting to true;
- with per-lane convergence enabled, freeze and remove a lane after a complete Jacobi round produces no change for that lane;
- preserve the two-buffer equality invariant for a converged lane so later batch-wide buffer swaps are safe;
- width 1 agrees exactly with standalone Jacobi;
- every lane agrees with an independent bounded Dijkstra query;
- support all three control modes;
- persistent convergence remains GPU-controlled;
- reached-target and miss masks are computed on the GPU after convergence.

Add mixed-duration batch tests in which different lanes converge immediately, after one round, after several rounds, and as unreachable bounded queries. Run them with per-lane convergence enabled and disabled in every control mode. Include a case where one lane converges before several later Jacobi slot swaps and verify that both of its distance-buffer columns remain bitwise identical and correct.

Measure:

- edge reuse;
- lane utilization;
- inactive/padded work;
- work avoided by per-lane convergence and tail rounds remaining after each lane converges;
- CSC runs visited, skipped, and average active lanes per visited run;
- union-tile inflation;
- buffer reset traffic;
- register pressure;
- occupancy;
- L2/read traffic;
- per-query latency and batch throughput.

Acceptance:

- all lanes correct;
- all controls correct;
- enabled and disabled per-lane convergence produce identical final distances;
- batching improves or honestly fails to improve throughput;
- no claim relies only on CPU batching estimates.

Stop after Phase 14.

---

# Phase 15 — overlapping batched dense chaotic push

Implement batched `DenseChaoticPush` for widths 1, 8, 16, and 32.

Traverse outgoing CSR destination-tile runs. Compute the active lane mask once per run, skip zero-mask runs, and reuse the mask for all edges in the run.

Prefer one wave per admitted edge, with lanes representing queries, when that mapping is supported by measurements.

Each lane:

- atomically loads its source distance;
- adds the shared edge weight;
- atomically decreases its independent destination-lane distance;
- updates lane-specific changed state.

Requirements:

- separate lane distance state;
- no cross-lane source merging;
- no frontier;
- support `enable_per_lane_convergence=true|false`, defaulting to true;
- with per-lane convergence enabled, freeze and remove a lane only after one complete admitted-edge scan produces no strict decrease for that lane;
- all three control modes;
- width 1 agreement with standalone dense chaotic push;
- per-lane bounded Dijkstra agreement.

Add mixed-duration batch tests in which different lanes require different numbers of complete admitted-edge scans, including a lane with no first-round decrease and a bounded unreachable lane. Run them with per-lane convergence enabled and disabled in every control mode.

Measure:

- atomic attempts per useful update;
- lane occupancy;
- work avoided by per-lane convergence and tail rounds remaining after each lane converges;
- CSR runs visited, skipped, and average active lanes per visited run;
- shared edge-load effectiveness;
- L2 writes and write stalls;
- register pressure;
- occupancy;
- full-edge rounds;
- throughput relative to batched Jacobi.

Acceptance:

- every lane correct;
- no cross-lane contamination;
- enabled and disabled per-lane convergence produce identical final distances;
- atomic-compatible loads preserved;
- batching behavior is profiled independently of standalone behavior.

Stop after Phase 15.

---

# Phase 16 — overlapping batched frontier push

Implement batched `FrontierPush` for widths 1, 8, 16, and 32.

Represent per-vertex activity with a wave32 lane mask. A worklist entry represents a vertex active for at least one lane.

Use the simple vertex-frontier scheduling policy fixed in Phase 11. Traverse each active vertex's outgoing CSR destination-tile runs, compute admission once per run, and reuse that run mask for its edges.

For an active vertex:

- read its current active-lane mask;
- traverse each admitted outgoing edge once;
- relax only lanes admitted for both endpoint tiles;
- atomically OR successful lanes into the destination’s next mask;
- append the destination vertex only on the zero-to-nonzero next-mask transition.

Requirements:

- preserve lane-specific distances;
- maintain vertex-level queue deduplication;
- allow multiple successful lanes to share one queue entry;
- explicitly detect queue overflow;
- support `enable_per_lane_convergence=true|false`, defaulting to true;
- with per-lane convergence enabled, freeze and remove a lane when it contributes no entry to the complete next frontier;
- when per-lane convergence is disabled, do not invent frontier work for an absent lane; terminate only when the complete batch frontier is empty;
- support all three control modes;
- width 1 agreement with standalone frontier push;
- independent bounded Dijkstra agreement for every lane.

Add mixed-duration batch tests in which lane frontiers empty in different rounds, including a lane with no next-frontier entry while other lanes continue and a bounded unreachable lane. Run them with per-lane convergence enabled and disabled in every control mode.

Measure:

- vertices and edges shared across lanes;
- active lanes per worklist vertex;
- queue entries saved by lane merging;
- mask atomic operations;
- distance atomic attempts/successes;
- duplicate suppressions;
- queue high-water mark;
- lane utilization;
- work avoided by per-lane convergence and tail rounds remaining after each lane converges;
- CSR runs visited, skipped, and average active lanes per visited run;
- occupancy and memory traffic;
- throughput relative to the other batched engines.

Acceptance:

- every lane correct;
- queue and mask invariants pass;
- enabled and disabled per-lane convergence produce identical final distances;
- no host frontier polling in persistent mode;
- no silent fallback to dense chaotic push;
- edge balancing, wave aggregation, and adaptive frontier scheduling remain documented future work and are not implemented in this phase.

Stop after Phase 16.

---

# Phase 17 — batched bounding-box expansion and all-query execution

Implement reachability-triggered batched expansion across all three batched engines.

After one batch finishes:

- compute a per-lane all-targets-reached mask on the GPU;
- copy one compact batch status;
- accept reached lanes;
- collect failed query IDs;
- expand their tile regions;
- deterministically replan failed queries into new overlap batches;
- restart failed queries from their original source sets.

Evaluate expansion schedules:

```text
one geometric tile ring
fixed larger ring
doubling x/y margins
hybrid small-first then doubling
```

Measure retries and total work. Select a default from evidence, not intuition.

Required safety:

- preserve query identity across replanning;
- increment an expansion generation;
- never reuse stale distances after restart;
- include long cross-tile adjacency and spill behavior;
- apply a configurable maximum expansion count;
- use a final full-region fallback or explicit terminal failure;
- never report a missed target as success.

Tests:

- multiple misses in one batch;
- only one failed lane;
- failed lanes that become dissimilar after expansion;
- long edge reaching beyond one ring;
- spill-tile target;
- success after several expansions;
- final fallback;
- deterministic replan order;
- agreement with Dijkstra on each final admitted region.

Run all `logicnets_jscl` queries first. Then run the remaining contest benchmarks as resource limits permit.

Report:

- initial success rate;
- expansion count distribution;
- failed-lane batch utilization;
- replanning time;
- repeated edge work;
- final selected-region size;
- unreachable/fallback count;
- total all-query throughput.

Do not implement an exit lower-bound certificate. Do not expand a query whose targets are already reached.

Stop after Phase 17.

---

# Phase 18 — compact target/path results and end-to-end no-congestion campaign

Add compact result production without adding congestion.

First copy only:

- per-target distance;
- reached status;
- selected source;
- path length or reconstruction status.

Then implement GPU post-relaxation path reconstruction from final distances and incoming CSC, independently extending the accepted Phase 5 semantics. Preserve stable-edge-ID tie-breaking and the Phase 5 zero-weight-cycle/backtracking semantics. Do not copy or mechanically transform historical reconstruction code.

Do not copy entire lane-distance matrices to the CPU in the timed production path.

Validate every sampled compact path for:

- source termination;
- target termination;
- edge continuity;
- exact tightness relative to the GPU distances;
- reported cost;
- bounding-region membership.

Run a complete no-congestion pipeline:

```text
load preprocessed graph/query artifacts
upload graph once
plan all query batches
run selected SSSP engine/control mode
perform batched expansion
extract compact target paths
return compact results
```

Report separately:

- cold artifact load;
- graph upload;
- batch planning;
- SSSP;
- expansion;
- reconstruction;
- result transfer;
- total warm all-query time;
- total cold pipeline time.

Run every engine/control combination on representative subsets. Run the recommended configuration over all available benchmark queries.

Report path-cost or path-length inflation against unbounded Dijkstra on a statistically documented sample. This is a quality metric, not a correctness failure when the bounded path is optimal in its admitted region.

Acceptance:

- no congestion or resource-conflict logic;
- compact result transfer;
- sampled paths validate;
- all requested queries receive a result or explicit terminal failure;
- end-to-end timing includes every stage claimed.

Stop after Phase 18.

---

# Phase 19 — final audit and evidence-based recommendation

Do not add algorithms.

Audit:

- three standalone engines;
- three control modes per engine;
- three independently batched engines;
- batching widths;
- batched expansion;
- compact reconstruction;
- all profiler groups;
- all correctness evidence;
- all performance claims.

Classify every feature as:

1. implemented and correctness-tested;
2. implemented and performance-profiled;
3. implemented but not yet representative;
4. designed but deferred.

Produce a final comparison including:

- best control mode per engine;
- best K per ordinary engine;
- best persistent grid policy;
- best batching width;
- best overlap-planner thresholds;
- best expansion schedule;
- P50/P95/P99 latency;
- all-query throughput;
- end-to-end time;
- box-miss and expansion rates;
- path-quality sample;
- memory footprint;
- synchronization and copy counts;
- profiler-supported bottleneck conclusions.

Do not create an adaptive hybrid. Recommend whether a hybrid should be a future experiment only after the standalone results are clear.

State explicitly:

- whether Jacobi min-plus pull benefits from the GPU;
- whether eliminating atomics outweighs dense edge scans;
- whether chaotic propagation reduces enough rounds to justify atomics;
- whether frontier work reduction offsets queue overhead;
- whether cooperative persistence beats chunking;
- whether overlapping batching is necessary for adequate utilization;
- which tail-query class dominates all-query runtime.

Stop after Phase 19.

---

# Start instruction

Execute Phase 6 only.

Do not begin Phase 7 during the same task, even if Phase 6 completes early.
