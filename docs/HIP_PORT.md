# HIP/ROCm Port

## Current status

HIP remains optional and disabled by default. Phase 6 provides the build and
measurement gate. Phase 8 adds checked device graph/workspace views and
retained resident ownership. Phases 9 and 10 add separately selectable,
HIP-gated Jacobi pull and dense chaotic-push SSSP sources. Phase 11 adds the
separately selectable active-frontier CSR-push source with bounded double
queues and per-generation deduplication. The portable models and structural
checks are CPU-only evidence; none of the three HIP engines has been compiled
or executed on a GPU locally. Phase 12 adds the host shootout schema, manifest,
schedule, evidence gates, and report path plus a HIP+FPGAIF-gated future
campaign driver. The driver has not been compiled with HIP or run, the real
query artifact is absent, and no representative timing or profiler evidence
exists. Phase 13 adds CPU-only overlapping-batch plans, batch-device
descriptions, exact tile/run lane masks, and workspace estimates. It adds no
HIP translation unit, device allocation, or batched kernel. Its provisional
full-vertex/retained-mask choice is not GPU-validated. Phase 14 adds a separate
HIP-gated batched Jacobi engine and reusable workspace for widths
1/8/16/32. The source implements wave32 ordinary and cooperative-persistent
paths, but it has not been compiled by a HIP compiler or executed locally.
Phase 15 adds a distinct HIP-gated `BatchedDenseChaoticPushEngine` and
`ReusableBatchedDenseWorkspace` for the same widths over retained CSR masks and
one in-place atomic word per vertex/lane. Neither Phase 14 nor Phase 15 HIP
source has been compiled by a HIP compiler or executed locally. Phase 16 adds
the separately gated `BatchedFrontierPushEngine` and
`ReusableBatchedFrontierWorkspace`. That workspace adds two vertex queues and
two graph-sized activity-mask arrays to the one-slot lane-private distance
state. It has likewise not
been compiled by a HIP compiler or executed locally. Portable CPU tests and
source inspection are not device-correctness or performance evidence. Phase
17 adds one HIP-gated sequential expansion executor above all three batch
engines. Its compact path reuses each engine finalizer and downloads one
48-byte status per batch; its evidence path reuses the existing status-only
transfers. Strict fake-HIP syntax is the only local device-source evidence for
that adapter. When both HIP and FPGAIF are enabled, the separate
`bfnew_gpu_batched_expansion` target provides the deferred all-query artifact
driver. It exists in source but has not been HIP-compiled or run locally.
Phase 18 adds the optional compact reconstruction path and repeated
no-congestion artifact campaign boundary; these sources likewise have only
strict source-level evidence locally. Phase 19 adds no HIP source or kernel. It
audits the complete optional surface and records zero performance-profiled SSSP
features, no measured device comparison, and no production recommendation.

The default CMake configuration is CPU-only. The following generic HIP build
is reference material for the combined campaign; do not run it during the
remaining implementation phases. On the ROCm target, enable the gate explicitly:

```bash
cmake -S . -B build-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip --parallel
ctest --test-dir build-hip --output-on-failure
```

An explicit HIP request fails during configuration if CMake is too old or no
HIP compiler is available. CPU-only builds neither detect nor link ROCm.

The configured FPGA24 data root contains
`logicnets_jscl_unrouted.phys` and `logicnets_jscl.netlist`, but those are
artifact-generation inputs rather than direct solver inputs. The required
versioned `logicnets_jscl.padding1.v1.bfqueries` artifact remains absent.

## Validated target contract

The campaign target is an AUP Radeon `gfx1151` system with wave32, 40 physical
CUs, 20 HIP multiprocessors/WGPs, cooperative launch support, ROCm 7.13.0, and
ROCprofiler-SDK 1.3.0. `hipDeviceProp_t::multiProcessorCount` is 20 in WGP mode;
that is the multiplier used for cooperative grid residency. The known 160-block
result belongs only to an earlier trivial probe. The current probe records its
own 256-thread occupancy, and every real kernel must query
`hipOccupancyMaxActiveBlocksPerMultiprocessor` for its own function, block size,
and dynamic shared-memory use.

The probe launches a real cooperative grid with two `this_grid().sync()` calls
and verifies state produced on both sides of the barriers. The barrier benchmark
keeps all workgroups in every grid barrier and queries occupancy separately for
each block size. Its reported amortized time includes kernel loop and launch
effects; it is an isolated control-cost measurement and does not predict SSSP
performance.

## Deferred profiling run

GPU execution is not required for phase acceptance. After all implementation
phases are complete and the combined GPU campaign begins, a maintainer can run
the additional `gfx1151` gate from the project root on that target:

```bash
tools/run_phase6_profiler.sh build-hip build-hip/phase6-profile
```

The script places all generated evidence below the build tree. It uses the
required global device selector placement for the catalog and compatibility
commands (`rocprofv3-avail -d 0 ...`), recursively splits rejected candidate
groups, records the accepted groups in the capability JSON, runs a real counter
collection on a repeated arithmetic/memory kernel, and captures HIP, kernel,
and memory-copy traces. Profiler-instrumented timings are not ordinary
performance results.

If collected, accepted-group and barrier findings belong in
`docs/PROFILER_COUNTERS.md` and `docs/COOPERATIVE_BARRIER_RESULTS.md`. Their
absence is not an acceptance blocker, but unexecuted HIP code must remain
clearly labeled GPU-unvalidated. Phase-specific deferred commands and evidence
boundaries are in `docs/PHASE8_GPU_VALIDATION.md`,
`docs/PHASE9_GPU_VALIDATION.md`, `docs/PHASE10_GPU_VALIDATION.md`, and
`docs/PHASE11_GPU_VALIDATION.md`. The cross-engine correctness, timing, and
profiling sequence is in `docs/PHASE12_GPU_VALIDATION.md`. These commands are
reserved for the combined GPU campaign after all implementation phases, not
for acceptance of an implementation phase. Batched-engine procedures are in
`docs/PHASE14_GPU_VALIDATION.md`, `docs/PHASE15_GPU_VALIDATION.md`, and
`docs/PHASE16_GPU_VALIDATION.md`.

## Phase 11 frontier source boundary

`FrontierPushEngine` consumes the immutable resident CSR image and reusable
query workspace on one stream. Construction verifies graph/run shape and the
resident upload's deterministic two-word content fingerprint. That fingerprint
is an accidental same-image guard, not cryptographic integrity evidence.

The frontier scratch request is one 32-bit distance word per vertex, two
bounded 32-bit queues, and one aligned 64-bit enqueue-generation word per
vertex. A zero requested capacity reserves one queue entry per vertex. A
smaller explicit capacity is only an overflow-validation seam; a full claim
sets `queue_overflow` and cannot write out of bounds. For `V` vertices and
capacity `Q`, scratch is `4V + 8Q + alignment_padding + 8V` bytes; default
`Q = V` is `20V` plus at most four padding bytes. The current fixed ABI is
eight-byte aligned with `DeviceWorkStatistics` at 168 bytes and
`GpuSsspResult` at 224 bytes.

Ordinary source paths enqueue one initializer, one frontier-round/controller-
advance pair per requested round, and one final status kernel. Per-round mode
polls after one pair; chunked mode polls after a complete `K`-pair chunk for
`K = 2,4,8,16,32`. For `N` polls, physical controller copies are initial H2D,
`N` observation D2Hs, and terminal D2H (`N + 2`); host synchronizations are
`N + 1`: the polls plus one shared final result-transfer/lease-retirement
boundary. Timing-event elapsed queries add no synchronization. Persistent
mode is one cooperative kernel containing initialization, run-mask setup, all
queue rounds/swaps, and final status, with no host queue-size poll. Real
frontier-kernel occupancy bounds cooperative grids; the trivial Phase 6 probe
ceiling is not reused.

The source performs one grid-stride thread's work per frontier entry, traverses
admitted CSR runs, uses CAS-domain source loads and strict destination minima,
deduplicates next-frontier claims by atomic generation exchange, and reserves
queue slots atomically. None/Light/Debug control algorithm counters without
changing labels. HIP reports aggregate active frontier vertices and executed
rounds, not the portable model's exact per-round frontier-size sequence. The
event-reported `gpu_milliseconds` is a device-timeline span; in host-poll modes
it includes stream-idle observation/re-enqueue gaps and is not summed
kernel-active time.

No HIP compiler, GPU, real occupancy query, dependency trace, profiler counter,
timing distribution, bounded-real case, or full graph has validated this
source. All such evidence remains deferred until after all implementation
phases.

## Phase 12 shootout execution boundary

The host `engine_shootout` module does not launch kernels. It defines the
canonical engine/control/block/K/grid catalog, resolves it against actual
per-kernel occupancy limits, selects the fingerprinted representative manifest,
constructs deterministic interleaved schedules, validates stage-specific
samples, gates later evidence on complete correctness, and produces stable
TSV/JSON records. The host-only `bfnew_shootout_report` consumes those records
and emits a globally capped, identity-bound profile-case plan; an optional
conclusions TSV can supply all seven evidence-backed answers and configurable
defaults. `bfnew_shootout_profile_import` validates normalized trace/PMC values
from one row or a strict one-to-one case-ID map for a marker-delimited batch.
The optional `bfnew_gpu_shootout` driver is available only when both
`BFNEW_ENABLE_HIP` and `BFNEW_ENABLE_FPGAIF` are enabled, because representative
execution requires the three HIP engines and the versioned real workload
artifacts. It also exposes separate built-in sparse-wavefront and dense-frontier
synthetic modes. The default build remains independent of both dependencies.

`EngineShootoutExecutor` constructs exactly one Jacobi, dense-chaotic, and
frontier engine over the same resident graph, reusable workspace, and stream.
It uses selected-tile distance results only for correctness and the engines' status-only
entry points for warmup, timing, Debug counters, and profile replay. Timing
rejects any workspace allocation-event growth across the engine call and any
unexpected distance vector. Pilot selected-edge features come from a one-time
CSR tile-pair index, not one full graph scan per query. The real pilot uses a
domain-separated cheap-feature prescreen of at most four times the requested
count and runs status-only Jacobi only on that shortlist. Correctness copies and
oracles only selected tile/admitted CSR-run ranges. Profile replay performs one
unrecorded status-only warmup per case and brackets every measured case with
named marker kernels recorded in a range ledger.

One immutable device graph must be uploaded before timed queries and retained
across the interleaved matrix. Every configuration for one manifest query uses
the same graph fingerprint, weights, selected-tile mask, canonical source set,
and canonical target set. Persistent legality comes from
`hipOccupancyMaxActiveBlocksPerMultiprocessor` on each real engine kernel at
block sizes 128, 256, and 512. Occupancy-derived and requested fixed-residency
grids are separate configurations; illegal block/residency combinations remain
recorded with reasons. The trivial Phase 6 probe ceiling is never substituted.

The correctness path downloads only selected-tile distances after the engine
completes and compares them with selected-region bounded CPU Dijkstra. It groups configurations by query,
assigns ordinals in that actual execution order, and builds the induced oracle
exactly once per query. That transfer is outside every timing sample. The final timing path uses instrumentation None and downloads only the
common status/result record, so Jacobi's two distance columns are not charged a
different graph-sized result copy from the one-column push engines.
Preparation-event, SSSP-device-timeline, result-transfer-event, and end-to-end
wall values remain separate. For ordinary controls, the SSSP device timeline
includes stream-idle host polling gaps; profiler trace durations are required
for kernel-active comparisons.

Algorithm counters are collected in separate Debug executions and joined
to timing by fingerprint, query ID, and configuration ID. Debug kernel limits
are resolved independently, and unsupported configurations remain explicit
unavailable cells. HIP/kernel/copy traces and every compatible PMC group use
additional processes. L2, occupancy,
memory-unit, and instruction fields carry explicit unavailable or
not-applicable state until matching profiler evidence exists. Profiler runtime
is never included in ordinary P50/P95/P99 or throughput calculations.

No HIP compiler, device correctness run, runtime engine occupancy, real
1,000-query manifest, warmed timing sample, dependency trace, PMC, or measured
comparison exists yet. Consequently all seven Phase 12 questions and all
recommended defaults remain pending, with every toggle configurable. The exact
future evidence sequence and current artifact limitation are in
`docs/PHASE12_GPU_VALIDATION.md`.

## Phase 13 host/device boundary

Phase 13 deliberately stops before HIP. Its `BatchDeviceDescription` is an
owning host preparation image with fixed-width device-facing fields, not a
resident allocation or launcher. It contains per-lane QueryIds and generations,
offset source and target payloads, selected estimates, sorted union tiles,
dense tile-lane masks, selected vertex ranges, zero initial result masks, and
one exact run representation.

Retained storage keeps one CSR and one CSC lane-mask array and touched ledgers;
reuse clears only the previously nonzero positions. Compact run storage keeps
strictly ordered nonzero `(run_id,lane_mask)` records plus per-union-vertex
CSR/CSC offsets. Compact vertex storage has a dense per-tile bias image so a
selected global vertex maps to `global_vertex - bias[owner_tile]`. Cold
initialization and warmed clear/write traffic are explicit, and a reusable
description cannot switch run representations after its first preparation.
Both run formats compute admission once per immutable maximal run:

```text
CSR mask = source-owner-tile mask & destination-tile mask
CSC mask = destination-owner-tile mask & source-tile mask
```

The bounded CPU proof confirms the two images admit the same lane/edge pairs
as Phase 8 endpoint-by-endpoint admission. A future kernel may therefore skip
zero-mask runs and reuse a nonzero mask for every edge in its run. Phase 13
does not establish which representation is faster on `gfx1151`.

The workspace model evaluates full graph-sized vertex-major distance lanes and
compact union-tile distance lanes, crossed with both run formats. It reports
checked component bytes and preparation traffic before allocation. The
bounded-synthetic provisional selection is full vertex-major storage plus
retained run masks because it avoids global/local label mapping and is simpler
to reuse. Its max-CSR/CSC capacity and traffic calculation is for a workspace
bound to one active orientation; a future engine that switches orientation
must use separate accounting or a separate workspace. Production selection is
deferred: there is no real `.bfqueries`
artifact, real union/run distribution, runtime free-memory sample, HIP build,
mapping/reset trace, or batched kernel timing.

For the Phase 7 recorded `V = 28,226,432`, distance labels alone require
`V * width * slots * 4` bytes. At two slots this is 1,806,491,648 bytes for
width 8, 3,612,983,296 for width 16, and 7,225,966,592 for width 32. The
documented 64 GiB nominal / 8 GiB resident allowance / 4 GiB reserve scenario
is illustrative arithmetic only; it is not a `hipMemGetInfo` result, does not
include real run/metadata storage, and does not prove a concurrent allocation
count.

The deferred artifact and combined-GPU commands are recorded in
`docs/PHASE13_WORKSPACE_DECISION.md`.

## Phase 14 batched Jacobi host/device boundary

`hip::BatchedJacobiPullEngine` deliberately does not implement the single-query
`GpuSsspEngine::run(RouteQuery)` surface. Its input is one validated
`BatchDeviceDescription`, because a batch needs independent terminal offsets,
lane identity, selected ranges, a low-prefix validity mask, and CSC run
admission. Treating all flattened sources as one `RouteQuery` would merge
queries incorrectly.

`ReusableBatchedJacobiWorkspace` is additive to the Phase 8 standalone
workspace. It owns geometrically retained device buffers for sources, targets,
dense tile masks, a device-only dense CSC mask image,
source/target offsets,
selected ranges, packed range offsets, controller, status, optional common and
batch statistics, convergence rounds, and engine scratch. The scratch size is
checked as `vertex_count * lane_width * two slots * sizeof(float)`. A lease
binds all preparation, kernels, transfers, and retirement to one stream.
Destruction fences conservatively only if a lease or possibly in-flight work
remains.

Selected ranges, rather than a redundant union-tile upload, are authoritative
on device. Initialization assigns every controller field before its first read,
and finalization assigns the complete status record before transfer, so batch
preparation performs neither controller H2D nor status memset. With `N`
ordinary polls, full/evidence execution has `N + 1` controller D2Hs and `N + 1`
host synchronizations; the terminal transfer also retires the stream lease.

Initialization traverses packed selected ranges. Its physical wave lanes first
cooperate over each selected destination column's CSC runs and write the exact
destination/source-tile intersection mask, including zero masks. Selected
columns are disjoint, the next ordinary kernel (or the persistent grid barrier)
provides visibility, and stale nonselected entries are never read. Thus no
graph-wide run-mask clear or H2D mask image is issued per batch. Every query
lane's own source slice then controls zero seeding; all other semantic selected
cells begin at positive infinity in both slots. Padding and nonselected
distance cells are not cleared or read. Reported reset traffic is therefore
the actual selected lane/vertex cells in two slots, not a silent whole-scratch
clear.

The relaxation kernel maps one wave32 to one destination. The first
`lane_width` wave lanes represent query lanes, and lanes outside the selected
destination/run masks remain inactive. The runtime rejects a device whose
reported wave size is not 32, a block size that is not a multiple of 32, or a
block/grid configuration that exceeds actual-kernel occupancy. Each incoming
CSC run computes:

```text
active_run_mask = device_materialized_endpoint_mask
                & destination_execute_mask
```

A zero result skips every edge in the run. Otherwise each edge record supplies
its source and weight to the active query lanes, which read and write distinct
vertex-major slots. Changed bits are reduced by wave and published with a mask
OR; no distance atomic is used.

The default `BatchedJacobiLoadStrategy::compiler_uniform` leaves immutable run
masks, sources, and weights as compiler-visible uniform loads. The
`explicit_wave_broadcast` alternative loads them in wave lane zero and uses
`__shfl`. Both ordinary and persistent kernel variants exist so the comparison
can later use identical batch/control/instrumentation conditions. Explicit
broadcast is not the default and is not retained as an optimization until the
target-device experiment in `docs/PHASE14_GPU_VALIDATION.md` is complete.

Ordinary execution uses initialization followed by ordered round/advance
pairs, then a device finalizer. Per-round polling observes after one pair;
chunked polling submits the configured `K` pairs before observation.
Persistent execution uses one cooperative kernel containing selected-range
initialization, every round, controller advancement, and final status. Every
workgroup reaches the grid barriers around each transition. The persistent
driver performs no convergence copy or host synchronization before terminal
result transfer.

The final device status iterates each valid lane's target slice after normal
convergence. It sets reached only when all targets are finite and otherwise
sets miss. Maximum-round or error stops publish neither. Per-lane convergence
rounds are copied separately for tail accounting, while status retains the
shared final slot and controller masks.

Light/Debug batch counters separate run visits and skips, nonzero runs, shared
edge records, admitted lane-edge pairs, active lanes across nonzero runs,
active destination/lane evaluations, and decreases. Metrics derive exact lane
rounds, tail work, selected reset bytes, union inflation, and algorithmic byte
requests. Optional L2/instruction counters remain unavailable unless a
separate matching profiler record supplies them. Timing instrumentation and
profiler instrumentation are never combined into one performance sample.

No local Phase 14 command enables HIP. The source has not been compiled by a
HIP compiler, launched on `gfx1151`, checked against Dijkstra on a device,
timed, traced, or profiled. The missing `logicnets_jscl` `.bfqueries` artifact
also prevents a real overlapping-batch campaign. Exact later commands,
correctness gates, load-strategy comparison, and reporting requirements are in
`docs/PHASE14_GPU_VALIDATION.md`.

## Phase 15 batched dense host/device boundary

`hip::BatchedDenseChaoticPushEngine` is a batch-specific engine, not a
`GpuSsspEngine::run(RouteQuery)` adapter. It accepts one validated
`BatchDeviceDescription` through `run_status_only` or `run_with_distances`.
The flattened terminal arrays remain separated by lane offsets; treating them
as one query would merge independent source sets and corrupt semantics.

`ReusableBatchedDenseWorkspace` is distinct from both the standalone workspace
and the two-slot Jacobi batch workspace. It geometrically retains device
buffers for sources, targets, union tiles, dense tile masks, exact retained CSR
masks, source/target offsets, selected ranges, packed range offsets,
controller/status, optional common and batch statistics, per-lane convergence
rounds, and engine scratch. The scratch requirement is checked as
`vertex_count * lane_width * sizeof(uint32_t)`. One lease binds preparation,
ordinary or persistent execution, result transfer, and retirement to one
stream, with conservative recovery only when work may remain in flight.

Initialization traverses packed selected ranges. A wave lane writes only the
vertex/query word admitted by the range mask, using that query lane's own
source slice to select positive-zero rather than positive-infinity bits.
Nonselected and padded words are not cleared or read semantically. Reported
reset traffic is therefore one selected 32-bit word per admitted vertex/lane;
source-seed bytes identify a subset of those initialization writes rather than
additional traffic. Total retained capacity is not reset traffic.

The current HIP relaxation maps one wave32 to a selected source row. Query
identities occupy wave lanes, and the wave services each outgoing edge request
across the admitted query lanes before advancing through the row. For every
CSR destination-tile run it computes:

```text
active_run_mask = retained_endpoint_mask & execute_lane_mask
```

A zero mask skips the whole run. Otherwise the same mask is reused for every
edge. Each admitted lane reloads its own source word with the CAS-compatible
atomic load, adds the edge weight, and performs a strict atomic minimum on its
own destination word. This in-place chaotic scan has no ordinary distance read,
second slot, CSC traversal, frontier, queue, or target early stop.

`BatchedDenseLoadStrategy::compiler_uniform` is the default: retained masks,
destinations, and weights remain compiler-visible uniform loads.
`explicit_wave_broadcast` loads those immutable fields in wave lane zero and
uses wave shuffle/broadcast. Both ordinary and persistent kernel variants are
present solely for a later correctness-matched comparison. The broadcast path
is not selected and neither source shape establishes which path emits fewer
physical loads.

Per-round and chunked controls launch ordered dense-scan/controller-advance
pairs, followed by a device finalizer. Chunked control submits the configured
K before polling; queued pairs after `done` no-op. Persistent control uses one
cooperative kernel containing initialization, all complete scans, controller
advancement, and final status, with every workgroup reaching the grid barriers
around each transition. Launches reject non-wave32 devices, incompatible block
sizes, and grids exceeding actual selected-kernel residency. The Phase 6 probe
ceiling is not reused.

The device finalizer classifies each valid lane's own targets only after normal
convergence. Maximum-round and error exits produce neither reached nor miss.
Convergence rounds are copied separately for executed/tail accounting. The
downloaded correctness form preserves both float values and their authoritative
uint32 atomic words; the status-only form omits the graph-sized download.

Light/Debug statistics record considered/skipped/nonzero CSR runs, algorithmic
edge-record requests, lane-edge pairs, atomic source loads and min attempts,
useful updates, active source/lane evaluations, changed publications, and full
scans. Host reconstruction derives exact tails, early-convergence work avoided,
lane-width/wave32/edge-wave capacity, union inflation, and selected reset bytes.
These are logical/request counts. In particular, `csr_edge_loads` and
`edge_record_read_bytes_requested` are not measured physical cache loads or L2
traffic. Optional L2-read/L2-write/atomic-stall/write-stall fields remain
explicitly unavailable until a matching profiler run supplies them.

The common `high_contention_destinations` field is implemented for standalone
dense diagnostics only. Phase 15 batched dense does not compute it; its zero
value is unavailable, not a measured absence of contention. Physical atomic
or write-stall behavior remains a future profiler question.

The first HIP path deliberately uses the provisional Phase 13 full-vertex,
retained-CSR strategy. Portable compact-descriptor equivalence does not add a
compact HIP path or establish a production memory choice. The absent
`logicnets_jscl` `.bfqueries` artifact also prevents representative batch-plan,
workspace, correctness, and timing evidence.

No local Phase 15 command enables HIP. The translation units have not been
compiled with HIP, launched on `gfx1151`, compared against bounded Dijkstra on
device, timed, traced, or profiled. Register pressure, legal occupancy, L2
reads/writes, write/atomic stalls, latency, throughput, batching benefit, and
the uniform/broadcast decision are all unavailable. The later combined-
campaign commands and gates are in `docs/PHASE15_GPU_VALIDATION.md`.

Phase 15 itself does not implement frontier state, expansion/replanning,
reconstruction integration, congestion, adaptive selection, or a production
configuration. Phase 16 adds batched active-frontier push separately.

## Phase 16 batched active-frontier host/device boundary

`hip::BatchedFrontierPushEngine` is batch-specific rather than an adapter from
one `BatchDeviceDescription` to `GpuSsspEngine::run(RouteQuery)`. It consumes the
flattened terminals only through their per-lane offsets, preserving independent
source sets, target sets, selected tiles, and distance words. The status-only
path omits graph-sized label download; the correctness path preserves the
authoritative unsigned distance words and their float projection. Before any
launch, the production guard requires strictly increasing terminal slices,
unique valid-lane query IDs, canonical zero/invalid padding metadata, empty
padded terminal slices, and terminal owner tiles that admit their lane.

`ReusableBatchedFrontierWorkspace` is distinct from the standalone frontier,
batched Jacobi, and batched dense owners. It geometrically retains sources,
targets, union tiles, dense tile masks, exact retained CSR masks, source/target
offsets, selected ranges and packed range offsets, controller/status, optional
common and batch statistics, per-lane convergence rounds, and frontier engine
scratch. One lease binds preparation, execution, result transfer, and
retirement to one stream.

The checked scratch layout contains:

```text
one uint32 distance word per graph vertex/query lane
two bounded uint32 vertex queues
two LaneMask activity arrays with one word per graph vertex
```

A zero requested queue capacity selects `vertex_count`; an explicit capacity
must be in `[1, vertex_count]` and exists only for overflow validation. Queue
entries name vertices, not vertex/lane pairs. With the default capacity the
scratch formula is `4 * V * lane_width + 16 * V` bytes. The Phase 13 generic
one-slot estimate included the first term only and explicitly excluded queues;
it also did not include these activity masks. Therefore it is not a complete
Phase 16 allocation or free-memory result. The full-vertex/retained-CSR choice
remains provisional until the real corpus and target-device memory evidence
exist.

Initialization writes only selected semantic label cells and union-vertex
activity masks. Shared source vertices produce one initial queue entry carrying
the OR of their source-lane bits, while every source lane retains its own zero
distance word. Both activity-mask slots are cleared for every union vertex
before source masks are formed. Queue buffers themselves are read only below
their controller sizes and need no full clear. Ordinary initialization phases
are stream ordered, and a grid-stride loop resets every configured lane's
convergence record. Persistent initialization uses uniform cooperative-grid
barriers before any queue round can consume the state.

For full correctness readback, the host output is initialized to positive
infinity over the entire vertex-by-width shape and only selected semantic words
are overlaid from the downloaded scratch. Padded lanes and nonselected cells
therefore cannot expose retained stale values. This public-output
normalization is not a device write and is excluded from selected reset-traffic
accounting. The portable retained-mask and compact-descriptor paths are tested
for equality across every output word and terminal lane result; the HIP path
continues to accept retained CSR masks only.

The relaxation kernel intentionally keeps the Phase 11 one-thread-per-current-
entry scheduler. The owner reads and clears the current vertex mask. For each
outgoing CSR destination-tile run it intersects that mask with the controller
execute mask and the retained endpoint mask once, skips zero, and reuses a
nonzero result for every edge in the run. Each admitted lane uses an atomic-
compatible source load, ordinary float addition, and a strict destination
atomic minimum. Successful lane bits for one edge are atomically ORed into the
destination's next mask. Only an old mask of zero owns the atomic queue
reservation and write. A claim at or past capacity publishes `queue_overflow`
without writing out of bounds.

The controller advance is frontier-batch-specific. It consumes the complete
next queue and lane mask, increments only a real executed round, swaps queue
slots, clears the recycled size and next state, then selects explicit error,
empty-frontier convergence, exact maximum-round exhaustion, or continuation.
With per-lane convergence enabled, lanes absent from the complete next
frontier are marked converged. With it disabled, absent lanes still receive no
synthetic frontier work; only their controller-level converged bits wait until
the complete batch frontier is empty. Final distance bits are therefore
required to match between both settings. A lane convergence record is
published only after a clean accepted continue, convergence, or maximum-round
transition. Error, invalid-controller, and terminal no-op transitions publish
no convergence proof.

Ordinary per-round and chunked modes submit ordered round/advance pairs. A
complete configured K is submitted before each chunk observation, and pairs
after device `done` no-op. Persistent mode is one cooperative kernel containing
initialization, all queue work, controller advancement, and terminal status.
Every workgroup reaches every grid barrier, including terminal transitions;
there is no host frontier-size or convergence polling inside the persistent
loop. Legal residency must be queried from the actual selected frontier
kernel. The Phase 6 probe ceiling is never reused. Ordinary execution reports
the occupancy- and queue-work-capped launch count in
`BatchedFrontierRunMetrics::ordinary_grid_blocks`; the field is launch
metadata, not measured occupancy.

Instrumentation tiers are exact. `None` suppresses device algorithm counters.
`Light` separates physical worklist vertices from active vertex/lane pairs,
shared CSR edge-record requests from lane-edge relaxations, run visits/skips,
active lanes over nonzero runs, successful lane updates, and queue high-water.
`Debug` additionally exposes current/next/controller mask atomics, unique lane
activations, queue claims, entries saved by lane merging, same-lane and total
duplicate suppressions, and overflow events. Exact utilization/tail/reset terms
are algorithmic accounting. Requested bytes are not measured cache traffic.
Optional occupancy and L2/memory fields remain unavailable until a separate
matching target-device run supplies them; zero must not be interpreted as
measured zero.

The hot path retains its controller-shape, mask-admission, queue-capacity, and
out-of-bounds-write guards. The public HIP result and current deferred HIP
CTest do not expose a complete per-round queue-to-mask boundary ledger. The
portable model proves that ledger on bounded fixtures; equivalent device proof
requires a future or manual Debug diagnostic extension in the combined
campaign.

The implementation deliberately excludes prefix-sum edge balancing, virtual-
warp/high-degree scheduling, wave-aggregated queue insertion, and adaptive
frontier scheduling. These remain future optimization candidates. There is no
silent dense-chaotic fallback or adaptive engine switch.

Final local evidence is HIP-off. Release passed `17/17` in `1.56` seconds,
including Phase 16 `1/1` in `0.02` seconds. ASan+UBSan passed `17/17` in `3.03`
seconds, including Phase 16 `1/1` in `0.16` seconds. The deferred HIP test
passed strict-warnings host syntax and fake-HIP `__HIPCC__` syntax, including
its device probe. Phase 16 public headers passed strict host syntax, and both
production HIP translation units passed strict fake-HIP syntax. These checks
are not a HIP compiler invocation or device-correctness evidence. No browser,
cloud service, or large/full-graph test was used.

No local Phase 16 command enables HIP. The translation units have not been
compiled with a HIP compiler, launched on `gfx1151`, compared with bounded
Dijkstra on device, timed, traced, or profiled. The absent `logicnets_jscl`
query artifact also prevents representative plans and throughput comparisons.
Device correctness, actual occupancy, physical memory traffic, latency,
throughput, batching benefit, and comparison with batched Jacobi/dense are all
unavailable. The deferred combined-campaign procedure is
`docs/PHASE16_GPU_VALIDATION.md`.

Phase 16 itself does not add expansion, restart, or fallback. Phase 17 now
composes the three batch engines through the boundary below.

## Phase 17 compact expansion host/device boundary

`hip::BatchedExpansionExecutor` is a sequential adapter for the generic
`run_batched_expansion` controller. Its three constructor overloads bind
exactly one `BatchedJacobiPullEngine`, `BatchedDenseChaoticPushEngine`, or
`BatchedFrontierPushEngine` to the matching reusable workspace, immutable
resident graph, tile-run layout, stream, and run options. The binding rejects
an engine mismatch or invalid transfer/load strategy before execution. It is
noncopyable because one retained description/workspace lease is reused across
retry batches.

Every callback rebuilds `BatchDeviceDescription` from the retry-local QueryIds,
generations, selected tiles, terminals, and features. The first HIP path uses
the existing exact retained run masks. It then calls the chosen engine's
ordinary selected-only reset and source seed. This is the restart boundary:
no distance word, changed flag, queue, activity mask, controller field, or
convergence record from a failed generation is consumed by the next one.

All three Phase 14–16 finalizers already compute per-lane all-targets-reached
and bounding-region-miss masks. Phase 17 therefore adds no classification or
relaxation kernel. It adds two transfer modes:

- `compact_status` requires `InstrumentationLevel::none` and copies exactly
  one 48-byte `DeviceRunStatus` after finalization. It downloads no common or
  batch counter record, controller, per-lane convergence record, or distance
  image. Persistent control has one final compact status transfer; ordinary
  controls retain the controller copies required by their poll protocol.
- `status_and_work_evidence` uses the underlying engine's richer status-only
  path. Jacobi and dense report their shared/logical edge counts. Frontier
  reports those counts only when its instrumentation supplies them. Every
  other case is explicitly unavailable rather than a measured zero.

The host controller validates that a clean converged status exactly partitions
valid lanes. It removes reached queries, collects failed QueryIds only after
the pass, changes selected tiles and generation, and invokes the adapter again
after deterministic replanning. Maximum-round, overflow, invalid-controller,
and device-failure statuses classify no lane and cause no expansion. A
nonclean status with result bits is rejected as malformed.

A nonzero execution-configuration fingerprint is mandatory. It binds HIP
transfer/load choices that are not represented by the shared engine/planner
options and is folded into the schedule-comparison fingerprint. Evidence from
different runner configurations therefore cannot compare silently. The
four-schedule selector also requires a valid campaign and a unique best score;
it returns no schedule on a score tie. Per-query generation, retry/count, or
margin overflow becomes `identity_or_count_overflow` without aborting other
queries. Aggregate telemetry overflow is different: it aborts the campaign
fail-closed and never becomes a query disposition.

The HIP+FPGAIF-gated `bfnew_gpu_batched_expansion` target is the implemented
deferred corpus driver. Its actual command surface requires graph, query,
output, engine, and schedule choices and exposes the four schedules, terminal
policy, controller/grid/convergence settings, and compact/None or evidence/
Light-or-Debug transfer modes. It deeply validates the versioned artifacts,
uploads one resident graph, runs every query, validates canonical outcome and
trace ledgers, and publishes a non-overwriting `bfnew.batched-expansion.v1`
TSV through a temporary file and rename. The report includes input/device,
execution-configuration, and schedule-comparison fingerprints, full
configuration, metrics, histogram, outcomes, selected tiles, and retry traces.
Engine or per-query identity/count failures yield diagnostic evidence and a
nonzero exit; aggregate counter overflow aborts the campaign. The driver has
no label-download/Dijkstra mode, statistical repetition aggregation, or
independent kernel/transfer timing. It synchronizes resident upload before the
controller, then starts with a cold reusable workspace and reuses that capacity
only inside the one campaign. Its TSV explicitly records
`resident_graph_upload=completed-before-controller`,
`workspace_initial_state=cold-reused-within-campaign`, and
`timing_boundary=controller-including-first-workspace-growth`. The resulting
throughput is correctness/diagnostic cold-controller evidence, not a warm final
measurement; final performance needs a later in-process repeated pass with a
pre-grown workspace.

The deferred device test covers all four schedules and all three executor/
workspace bindings in compact mode, repeated workspace reuse, multiple misses,
dissimilar regions, long and spill paths, retry identities/generations,
full-region fallback, a frontier one-miss compaction, persistent compact
execution, Debug evidence transfer, compact/instrumentation rejection, and
maximum-round no-expansion. It passes strict host and fake-HIP syntax only.
Those checks do not compile with HIP or prove device distances; the combined
campaign must still compare final admitted regions with Dijkstra as specified
by `docs/PHASE17_GPU_VALIDATION.md`.

No local Phase 17 command was run with HIP or used the absent
`logicnets_jscl.padding1.v1.bfqueries` artifact. Real device correctness,
48-byte transfer traces, restart isolation, actual occupancy, physical memory
traffic, replanning and transfer latency, all-query throughput, retry work,
and schedule selection remain unavailable. No schedule is a production
default.

Phase 17 stops before compact target/path output and reconstruction integration.
Phase 18 now adds the separate post-finalizer boundary below.

## Phase 18 compact reconstruction host/device boundary

### Resident reconstruction identity

The relaxation-hot graph previously omitted logical edge identity. Phase 18
adds one checked 32-bit `csc_edge_ids` word per incoming CSC record because
stable-edge tie-breaking is observable reconstruction behavior. The graph is
already device-representable only when its edge count fits the 32-bit ABI, so
every 64-bit host `EdgeId` narrowing is checked and IDs remain lossless for the
resident image. Layout validation compares each word with the host incoming
view. Memory reporting, graph fingerprints, upload, download, and resident
round-trip validation all include this component. CSR still carries no edge-ID
field because reconstruction walks incoming CSC only.

`DeviceCscView32::edge_ids` exposes the immutable resident array. It is not
read or written by hot relaxation kernels and adds no predecessor state to
their workspaces.

### Final-distance and batch views

`DeviceCompactDistanceMatrix` is a non-owning view of the exact final lane
matrix. It contains vertex count, lane width, slot count/stride, an explicit
encoding, and exactly one typed data pointer:

- Jacobi supplies both contiguous floating-point slots and the device worker
  resolves `DeviceRunStatus::final_distance_slot` before reading a label,
  avoiding a preliminary status transfer; or
- dense/frontier supply their one-slot nonnegative-float bit words.

`DeviceCompactBatchView` supplies lane width, valid lanes, and device source/
target offsets. Sources, targets, tile-lane masks, and the final status remain
owned by the active engine workspace. Both views are fixed-layout,
trivially-copyable kernel argument records. They remain valid only until the
engine lease is retired or reused.

### Two-pass compact workspace

`hip::ReusableCompactPathWorkspace` owns the post-relaxation scratch and
result buffers independently of all three engine workspaces. It is noncopyable
and reports device capacity, DFS vertex capacity, target capacity, and
allocation-event count. One deterministic device worker processes targets
lane-by-lane and reuses a single `O(V)` DFS/backtracking stack rather than
allocating a predecessor or stack image per target.

Pass one reads the final status, labels, selected tile masks, canonical source/
target slices, incoming CSC, and stable edge IDs. It writes one fixed 28-byte
`CompactTargetSummary` per flattened target. Tightness is exact ordinary float
addition/equality. Candidate selection follows increasing stable edge ID,
checks path-local membership, and truly backtracks from cycles or dead ends.
Every clean valid lane receives explicit summaries; padding receives none.
Reached lanes must be wholly complete. Miss lanes may contain complete and
unreachable targets, while an error batch publishes no compact payload.

After the first mandatory summary transfer and synchronization, the host
validates statuses and performs checked deterministic prefixing of path
lengths. It grows exact compact arenas and launches pass two. The second pass
repeats the deterministic traversal and emits:

```text
path_length + 1 checked-u32 vertex IDs
path_length + 1 actual final float labels
path_length checked-u32 stable edge IDs
```

Only those used prefixes are copied. Host materialization widens device edge
IDs back to strong 64-bit `EdgeId` values. No `vertex_count * lane_width`
distance image crosses the device boundary. The aligned path labels are enough
for ordinary host validation to prove every returned tight edge against the
actual final GPU values.

`CompactPathBatchOutput` contains final status, flattened target results,
checked transfer accounting, and separate SSSP-device, reconstruction-device,
result-transfer-device, and end-to-end wall observations. The device-event
fields are stage spans, not host time or summed kernel-active time. The method
synchronizes only the two mandatory compact transfer boundaries in addition to
any controller dependencies required earlier by the selected control mode.
`CompactPathTransportAccounting` reports the fixed status and reconstruction-
error words separately and checks their sum with the public compact arenas as
the compact-result D2H subtotal. It is not the whole control-mode envelope:
per-round and chunked execution report controller-poll count/bytes separately,
persistent execution reports zero polls, poll bytes equal the count times the
96-byte `DeviceController`, and the checked overall D2H total adds poll bytes
to the compact subtotal.

### Engine/expansion integration and safety

The reconstruction call occurs after the existing engine finalizer has
materialized a clean status and before its workspace retires. Phase 18 does not
change relaxation, target classification, or restart kernels. All three
engines expose `run_compact_paths(...)`. The Phase 17 executor's
`compact_paths` transfer mode owns one reusable reconstruction workspace and
returns one canonical, generation-matched payload per clean valid lane. A
reached payload is captured immediately and must be wholly complete. A retry
miss is discarded with its generation only after every target is complete or
unreachable and at least one target is unreachable. An all-complete miss or a
miss containing a reconstruction failure becomes explicit engine failure
instead of entering retry. A terminal full-region miss retains its complete/
unreachable summaries. Padded, missing, duplicate, stale, or wrong-generation
payloads fail closed, and reached status with an incomplete payload becomes
explicit engine failure.

`CompactTransferAccounting` separates summary, vertex, path-label, edge-ID,
and total bytes. Trace evidence must match those values and must reject any
graph-sized D2H label copy in a production pass. A separate bounded diagnostic
may still download terminal labels for full Dijkstra comparison, but that copy
is not timed production execution.

The portable no-congestion final-serialization model is not this physical
ledger: it excludes retry payloads and leaves device evidence unavailable. A
HIP campaign must populate the separate cumulative compact payload/status/error
subtotal, controller-poll count/bytes, and overall D2H fields, including retry
summaries and polls, before making a D2H claim.

The settled optional surface is
`include/bfnew/hip/compact_path_results.hpp`,
`src/hip/compact_path_results.hip.cpp`, and
`tests/compact_path_results_test.hip.cpp`. The existing HIP+FPGAIF campaign
driver adds explicit `--transfer paths`, independent compact validation,
no-congestion target rows, and separate stage/byte ledgers. The timing schema
requires `warm_all_query` to equal the exact pre-grown named-host-stage sum
when every component is measured. When asynchronous execution cannot partition
a named host interval, that stage remains unavailable rather than measured
zero, and a separately measured enclosing `warm_all_query` supplies the warm
observation. The first capacity-growing `cold_execution` remains an independent
interval, and `cold_pipeline` is artifact load plus graph upload plus that cold
execution; cold and warm execution are never equated. The optional HIP
implementation and its bounded device test have not been
compiled by a HIP compiler or executed locally. Strict host source/public-
header syntax and fake-`__HIPCC__` production, CLI, and deferred-test syntax
passed warning-clean, but remain source evidence only. Device correctness,
actual copy traffic, reconstruction latency, capacity behavior, and all-query
throughput are deferred to `docs/PHASE18_GPU_VALIDATION.md`.

The path campaign admits widths 1/8/16/32, using width 1 as a scalar baseline
with identical result semantics. Its evidence gate requires measured GPU-event
timing for the cold execution, every additional warmup, and every measured
repetition. Unavailable device timing aborts the campaign instead of becoming a
zero-duration stage or an omitted run.

Phase 18 stops before congestion/resource-conflict updates, an exit lower-
bound certificate, and adaptive engine switching. Phase 19 now supplies the
separate final audit.

## Phase 19 HIP evidence disposition

Phase 19 changes no HIP translation unit and launches no kernel. Its canonical
local report classifies the optional standalone engines, batched engines,
expansion adapter, compact reconstruction, and artifact driver as implemented
without representative device evidence. The corresponding portable semantic
contracts remain separately correctness-tested.

No HIP compiler/runtime/device run, bounded HIP CTest, real-corpus timing,
marker-delimited dependency trace, compatible PMC group, physical peak-memory
observation, copy/synchronization trace, path-quality campaign, or normalized
per-query/class tail attribution is present.
Consequently every device-performance comparison and all seven Phase 19
questions remain insufficient-evidence. No engine, control, `K`, grid, width,
planner threshold, expansion schedule, or production configuration is selected.

This status is a completed audit result, not a GPU-validation claim. A future
maintainer campaign may follow `docs/PHASE12_GPU_VALIDATION.md`,
`docs/PHASE17_GPU_VALIDATION.md`, and `docs/PHASE18_GPU_VALIDATION.md`, but its
outputs must pass the Phase 19 representative-evidence gates before changing
any conclusion. Phase 19 adds no adaptive hybrid or congestion logic.
