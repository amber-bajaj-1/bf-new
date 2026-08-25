# Deferred Phase 15 batched-dense GPU validation

## Status and execution policy

Do not run these commands during the implementation phases. This runbook is
for the combined GPU campaign after all implementation phases are complete and
the user explicitly starts that campaign. It does not authorize browser use,
access to `sjc.aupcloud`, GPU testing by the coding agent, or a local full-graph
CPU run.

Phase 15 local acceptance consists of bounded portable semantic/controller
tests and structural inspection of the HIP-gated source. The final Release
`build/phase15-cpu` suite passed `16/16` CTests in `0.17` seconds; the focused
`bfnew.batched_dense_chaotic_push` CTest passed `1/1` in `0.01` seconds; and
the ASan+UBSan `build/phase15-asan` suite passed `16/16` in `1.82` seconds,
including Phase 15 `1/1` in `0.14` seconds.

No HIP compiler has compiled the Phase 15 translation units, and no AMD GPU
has executed their ordinary or persistent kernels. Device correctness,
register pressure, actual occupancy, physical L2 reads/writes, atomic or write
stalls, latency, throughput, batching benefit, and compiler-uniform versus
explicit-broadcast evidence are all unavailable. No performance claim or
production configuration follows from the source or portable counters.

The first combined-campaign target remains the previously recorded AUP
`gfx1151` environment:

- wave size 32;
- 20 runtime-reported HIP multiprocessors/WGPs and 40 architectural CUs;
- cooperative launch support;
- ROCm 7.13.0; and
- ROCprofiler-SDK 1.3.0.

Record the values actually reported during the future run. A mismatch is a
gate, not permission to substitute these expected values silently.

## Implemented boundary to validate

The public device surface is `hip::BatchedDenseChaoticPushEngine`, using
`ReusableBatchedDenseWorkspace`, `run_status_only`, `run_with_distances`, and
`batched_dense_distance_scratch_bytes`. The selectable load strategies are
`BatchedDenseLoadStrategy::compiler_uniform` and
`BatchedDenseLoadStrategy::explicit_wave_broadcast`; compiler-uniform is only
the unmeasured default.

The initial device implementation intentionally consumes the provisional
Phase 13 full graph-sized, vertex-major and retained-CSR layout. It retains one
unsigned 32-bit distance word per vertex/configured lane. Portable retained-
mask/compact-descriptor parity does not implement compact descriptors on the
device and does not promote a compact production layout.

A wave32 maps to one selected source row. Query identities occupy wave lanes,
and the wave services that row's outgoing CSR edge requests across admitted
query lanes. This mapping must be validated for widths 1, 8, 16, and 32; width
one is the singleton standalone-comparison path, while 32/16/8 remains the
standard overlap-plan family.

Preparation must initialize exactly one selected distance word for each
selected vertex/lane pair before relaxation, then seed the source subset with
positive zero. Total retained scratch capacity, padded lanes, and nonselected
vertices are not reset traffic. Sources, targets, changed bits, convergence,
status masks, and atomic destination words remain lane-independent.

## Required evidence identities

Every retained row must identify:

- resident graph fingerprint;
- query-artifact fingerprint and query count;
- batch-plan fingerprint, width, batch ordinal, valid mask, and QueryIds by
  lane;
- control mode and `K` for chunked control;
- block size, grid policy, and fixed blocks per WGP where applicable;
- per-lane-convergence setting;
- compiler-uniform or explicit-wave-broadcast strategy;
- instrumentation/profiler stage;
- status-only or full-distance result mode;
- warmup and retained-repetition counts, interleaving seed, and execution
  ordinal; and
- software/device metadata, including source identity, CMake configuration,
  HIP compiler/runtime, device name, architecture, wave size, WGP count, and
  cooperative-launch support.

Never merge rows whose identities differ. Artifact loading, CPU oracle time,
planning, preparation, correctness downloads, profiler overhead, and timed
status-only execution are separate stages.

## Real-workload prerequisite

The required real query artifact is absent. The interrupted Phase 7 run did
not emit:

```text
out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries
```

The long FPGA Interchange command and completion gates are documented in
`docs/PHASE12_GPU_VALIDATION.md`, section “Complete the versioned real-workload
artifacts”, and in `docs/PHASE13_WORKSPACE_DECISION.md`. Run that prerequisite
only during the combined campaign. Require graph/query fingerprints, artifact
round trip, all-input unchanged proof, query validation, and the bounded
Dijkstra sample to pass. A partial artifact is not usable Phase 15 evidence.

The tiny device gate below may run before that long prerequisite. It does not
establish representative correctness or performance.

## 1. Fresh HIP build and test inventory

Run from the `bf-new` root on the validated target. Preserve configuration and
build logs:

```bash
cmake -S . -B build/phase15-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  2>&1 | tee build/phase15-hip-configure.log
cmake --build build/phase15-hip --parallel \
  2>&1 | tee build/phase15-hip-build.log
ctest --test-dir build/phase15-hip -N \
  2>&1 | tee build/phase15-hip-tests.txt
```

Confirm that the inventory contains the portable
`bfnew.batched_dense_chaotic_push` CTest from target
`bfnew_batched_dense_chaotic_push_test` and the HIP-gated
`bfnew.batched_dense_chaotic_push_gpu` CTest from target
`bfnew_batched_dense_chaotic_push_hip_test`. The HIP CTest's 240-second
timeout is a safety guard, not a performance measurement.

The configuration must fail clearly if HIP was requested but unavailable.
Warnings, source guards, fake declarations, or a host compiler accepting
headers are not substitutes for compiling both Phase 15 `.hip.cpp`
translation units with the HIP compiler.

## 2. Tiny device-correctness gate

Run only the bounded portable and device Phase 15 tests:

```bash
ctest --test-dir build/phase15-hip \
  --output-on-failure \
  -R 'bfnew\.(batched_dense_chaotic_push|batched_dense_chaotic_push_gpu)$' \
  2>&1 | tee build/phase15-hip-correctness.log
```

Require the complete matrix before timing or profiling:

- widths 1, 8, 16, and 32;
- persistent cooperative and per-round host-poll controls;
- chunked host-poll control for `K = 2,4,8,16,32`;
- per-lane convergence enabled and disabled;
- compiler-uniform and explicit-wave-broadcast paths;
- None, Light, and Debug instrumentation preserving identical distance bits;
- independent and multi-source lanes without cross-lane seeding or updates;
- low-prefix padding contributing no source, target, selected, changed,
  status, atomic, or relaxation work;
- no-first-decrease, short, several-scan, and unreachable lanes;
- enabled and disabled convergence producing identical selected-region bits;
- width one agreeing bitwise with standalone dense chaotic push;
- every valid lane agreeing with its independent bounded Dijkstra oracle under
  the established exact/four-ULP fixture policy;
- reached and bounded-region-miss masks classified on device only after normal
  convergence;
- maximum-round and controller/device errors publishing neither reached nor
  miss; and
- fixed controller/status ABI validators and exact one-based convergence-round
  semantics.

Persistent evidence must show one cooperative convergence launch, uniform
grid barriers around every scan/controller transition, and no controller copy
or host synchronization inside its loop. Record actual selected-kernel
residency and legal grid size; never reuse a result from another engine or the
Phase 6 probe.

Any failure blocks later timing and profiling. Do not time or profile a
partial width, control, convergence, load-strategy, or instrumentation matrix.

## 3. Real plan and one-slot workspace gate

After the real artifact exists, rebuild deterministic Phase 13 features and
the 32/16/8 plan family. Validate once-only query assignment and exact graph,
query, and plan fingerprints before uploading any batch. Width one remains a
separate singleton baseline.

For each retained batch:

- deep-validate source/target offsets, union tiles, selected ranges and packed
  offsets, tile masks, retained CSR masks, and endpoint admission;
- require the retained-CSR representation and reject compact descriptors;
- compare checked component-byte calculations with retained device capacity;
- record allocation growth and warmed reuse;
- sample `hipMemGetInfo` after resident-graph upload and before each width
  reservation;
- reject widths or concurrency exceeding observed free memory after the
  resident graph and explicit reserve; and
- record selected one-slot reset bytes and source-seed bytes separately from
  allocated full-vertex distance capacity.

Do not reinterpret Phase 13's illustrative 64/8/4-GiB arithmetic as runtime
memory evidence. The workspace choice remains provisional until these real
observations exist.

## 4. Real-query correctness campaign

Freeze a fingerprinted query/batch manifest containing at least:

- one-source and multi-source queries;
- complete and padded batches at every standard width;
- low and high union-tile inflation;
- low and high active-lane retained-CSR masks;
- no-first-decrease, short, median, and long convergence durations; and
- reachable and bounded-region-miss lanes.

For every retained batch, run both convergence settings and every legal
control/K with compiler-uniform loads. Correctness runs use
`run_with_distances` outside timing and compare every selected lane/vertex word
with its own bounded Dijkstra result. Require label equivalence across controls
and convergence settings, valid reached/miss masks, no cross-lane
contamination, and padded semantic work exactly zero.

Run explicit wave broadcast through the identical correctness matrix before it
is eligible for timing. A single distance, status, convergence-round,
controller, or counter-semantics mismatch rejects that strategy.

## 5. Counter-only work characterization

Use separate Light and Debug runs, never the final timing samples. Record per
batch and aggregate distributions for:

- considered, visited, and skipped CSR runs;
- logical shared CSR edge-record requests;
- admitted lane-edge relaxations;
- atomic-compatible source loads, destination-min attempts, and useful
  updates;
- active lanes across visited runs and active source/lane evaluations;
- changed publications and complete edge scans;
- per-lane executed, convergence, tail, and avoided-tail scans;
- lane-edge work avoided by early convergence;
- union/selected tile-lane positions;
- selected one-slot reset and source-seed bytes; and
- exact valid-lane, configured-width, wave32, unused-wave, padded-lane, and
  edge-wave lane capacities.

`csr_edge_loads` is a logical shared edge-record request count. Requested
edge/source/destination bytes are likewise algorithmic requests. Neither is a
physical cache transaction, L2 measurement, or proof of compiler load
generation. The exact capacity terms are arithmetic denominators, not actual
occupancy.

`high_contention_destinations` is a standalone dense-engine diagnostic and is
not implemented for batched Phase 15. Its zero in the shared result structure
means unavailable, not a measured absence of contention. Measure physical
atomic/write stalls separately with compatible profiler counters.

## 6. Timing protocol

Only configurations passing the complete correctness gate are eligible. Use
instrumentation None and `run_status_only`. Perform at least five warmups and
thirty retained repetitions per configuration with a deterministic
interleaving seed across widths, controls/K values, convergence settings, and
load strategies. Do not give one strategy all warm-cache positions.

Record adjacent intervals for:

1. preparation and metadata transfer;
2. initialization through terminal status on the device timeline;
3. status/result transfer; and
4. end-to-end host wall time, including host polling where applicable.

For host-poll controls, the device-event span can include stream-idle gaps
while the host observes and re-enqueues work; it is not summed kernel-active
time. `batch_queries_per_second` is derived from that device-timeline span and
inherits the same limitation; final cross-configuration comparisons use the
retained wall-time samples. Use a trace for kernel-active duration. Full
distance downloads and CPU oracle work belong only to correctness and must
stay outside retained timing.

Report P50/P95/P99 batch wall and device-timeline time, explicitly defined
per-query latency, valid queries per second, and long-tail batch/QueryIds.
Compare widths 1/8/16/32; convergence enabled/disabled; persistent, per-round,
and all chunk K values; and compiler-uniform/explicit-broadcast. A batching or
throughput benefit may be claimed only when the full correctness-gated,
workload-matched distribution supports it.

## 7. Uniform-load versus broadcast decision

Compiler-uniform remains the default unless explicit broadcast passes the
same correctness matrix and shows a repeatable benefit on the identical real
manifest. Compare ordinary and persistent kernels separately and retain at
least:

- all widths and controls/K values;
- registers per thread;
- actual active blocks per WGP, cooperative legal grid, and achieved
  occupancy/residency;
- kernel-active and end-to-end timing distributions;
- physical L2 reads and writes, cache/load instruction evidence, and compatible
  atomic/write-stall counters; and
- effects on long-tail batches and total query throughput.

Do not retain explicit broadcast because source inspection appears favorable.
If results are noisy, mixed, workload-specific, or reduce legal residency,
keep compiler-uniform as the default and record broadcast as unselected.

## 8. Trace and profiler runs

Profiler runs use separate processes and the exact correctness-approved
configuration identity. First retain tool and counter availability:

```bash
mkdir -p build/phase15-hip/profiler
rocprofv3 --version \
  >build/phase15-hip/profiler/rocprofv3-version.txt 2>&1
rocprofv3-avail -d 0 list --pmc \
  >build/phase15-hip/profiler/pmc-list.txt 2>&1
rocprofv3-avail -d 0 info --pmc \
  >build/phase15-hip/profiler/pmc-info.txt 2>&1
```

Use the trace and PMC command form already validated for the target in
`docs/PHASE12_GPU_VALIDATION.md`. A bounded HIP correctness CTest is not a
throughput, interleaved timing, or profiler harness. If no reviewed combined
campaign driver exists, add one before measuring; do not invent CLI arguments
or treat a CTest loop as throughput evidence.

Collect compatible PMC groups in separate processes. Retain actual kernel
resource records, occupancy/residency, physical L2 reads and writes,
memory/atomic/write-stall activity, and load/instruction data needed for the
uniform/broadcast decision. Unavailable counters remain unavailable and are
never encoded as zero. Profiler runs remain outside ordinary timing
distributions.

## 9. Required final report

The Phase 15 GPU report must answer, with direct evidence references:

1. Do every width, lane, control/K, convergence setting, and load strategy
   remain correct without cross-lane contamination?
2. How much logical shared-edge work expands into lane-edge and atomic work?
3. What are valid/configured/wave32/edge-wave utilization, padding, skipped-
   run, and union-inflation distributions?
4. How much work does per-lane convergence avoid, and what does it cost?
5. What are actual selected one-slot reset traffic and retained workspace
   capacity under observed free memory?
6. What physical L2 reads/writes and atomic/write stalls occur? The logical
   request counters do not answer this question.
7. Does explicit broadcast beat compiler-uniform without harming registers,
   occupancy, or legal cooperative residency?
8. Which width/control/convergence configuration maximizes query throughput?
9. Does batching improve throughput over the width-one standalone semantic
   baseline?

Every conclusion names its graph/query/plan fingerprints, manifest, hardware,
software, configuration, timing stage, and profiler evidence. CPU estimates
and source inspection answer no performance question.

## Current conclusion

All Phase 15 device, correctness, register, occupancy, L2, write-stall,
latency, throughput, and uniform-versus-broadcast questions are pending.
Compiler-uniform remains the unmeasured default; explicit broadcast remains an
unselected experiment; the full-vertex/retained-CSR choice remains
provisional; and no batch width, control, convergence setting, performance
benefit, or production configuration is recommended. Phase 16 batched
frontier push is now implemented under the same CPU-only acceptance policy;
its separate deferred device procedure is `docs/PHASE16_GPU_VALIDATION.md`.
Phase 17 expansion/replanning is now implemented under the CPU-only acceptance
policy; its separate combined-campaign procedure is
`docs/PHASE17_GPU_VALIDATION.md`. This Phase 15 campaign stays deferred until
all implementation phases are complete.
