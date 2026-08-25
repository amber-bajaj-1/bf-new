# Deferred Phase 14 batched-Jacobi GPU validation

## Status and execution policy

Do not run these commands during the implementation phases. This is a runbook
for the combined GPU campaign after all implementation phases are complete and
the user explicitly starts that campaign. It does not authorize browser use,
access to `sjc.aupcloud`, GPU testing by the coding agent, or a local full-graph
CPU run.

Phase 14 local acceptance is bounded CPU semantic evidence and source-level HIP
evidence only. The Release `build/phase14-cpu` suite passed `15/15` CTests in
`1.38` seconds; its focused Phase 13+14 matrix passed `3/3` in `0.02` seconds,
and the Phase 14 CTest passed `1/1` in `0.01` seconds in the final full run. The
ASan+UBSan `build/phase14-asan` suite passed `15/15` in `2.73` seconds,
including the Phase 14 CTest `1/1` in `0.20` seconds. No HIP compiler has compiled
`src/hip/batched_jacobi.hip.cpp`, and no AMD GPU has executed its ordinary or
persistent kernels. Device correctness, actual occupancy, register pressure,
cache traffic, latency, throughput, and the compiler-uniform versus explicit-
broadcast comparison are all unavailable.

The first GPU campaign target remains the validated AUP `gfx1151` environment:

- wave size 32;
- 20 runtime-reported HIP multiprocessors/WGPs and 40 architectural CUs;
- cooperative launch support;
- ROCm 7.13.0; and
- ROCprofiler-SDK 1.3.0.

Record the actual values reported during the future run. A mismatch is a gate,
not permission to substitute these expected values silently.

## Required evidence identities

Every retained row must identify:

- the resident graph fingerprint;
- query-artifact fingerprint and query count;
- batch-plan fingerprint, width, batch ordinal, valid mask, and QueryIds by
  lane;
- control mode, `K` where applicable, block size, grid policy, and fixed
  blocks-per-WGP where applicable;
- per-lane-convergence setting;
- `compiler_uniform` or `explicit_wave_broadcast` load strategy;
- instrumentation/profiler stage;
- warmup/retained repetition counts and execution ordinal; and
- software/device metadata, including git state, CMake configuration, HIP
  compiler, runtime, device name, architecture, wave size, WGP count, and
  cooperative-launch support.

Never merge rows whose identities differ. CPU oracle time, artifact loading,
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
only during the combined campaign. Require the graph/query fingerprints,
artifact round trip, all-input unchanged proof, query validation, and bounded
Dijkstra sample to pass. A partial artifact is not usable Phase 14 evidence.

The tiny device tests below may run before that long prerequisite, but they do
not establish representative correctness or performance.

## 1. Fresh HIP build and test inventory

Run from the `bf-new` root on the validated target. Preserve configuration and
build logs under the build directory:

```bash
cmake -S . -B build/phase14-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  2>&1 | tee build/phase14-hip-configure.log
cmake --build build/phase14-hip --parallel \
  2>&1 | tee build/phase14-hip-build.log
ctest --test-dir build/phase14-hip -N \
  2>&1 | tee build/phase14-hip-tests.txt
```

Confirm that the inventory contains the portable `bfnew.batched_jacobi_pull`
CTest from target `bfnew_batched_jacobi_pull_test` and the HIP-gated
`bfnew.batched_jacobi` CTest from target `bfnew_batched_jacobi_hip_test`. The
HIP test has a 240-second CTest timeout; that guard is not a performance
measurement.

The configuration must fail clearly if HIP was requested but unavailable.
Warnings, fake declaration checks, or a host compiler accepting headers are
not substitutes for compiling the `.hip.cpp` translation unit with the HIP
compiler.

## 2. Tiny device-correctness gate

Run only the bounded device tests first:

```bash
ctest --test-dir build/phase14-hip \
  --output-on-failure \
  -R '^(bfnew\.batched_jacobi_pull|bfnew\.batched_jacobi)$' \
  2>&1 | tee build/phase14-hip-correctness.log
```

Require all of the following on device before any timing or profiling:

- widths 1, 8, 16, and 32;
- persistent cooperative, per-round polling, and chunked polling;
- per-lane convergence enabled and disabled;
- compiler-uniform and explicit-wave-broadcast load paths;
- independent and multi-source lanes with no cross-lane seeding;
- padding that contributes no source, target, status, changed, or relaxation
  bit;
- lanes converging immediately, after one changing round, after several
  rounds, and with unreachable targets;
- an early-frozen lane remaining bitwise identical in both selected-region
  columns through several later global slot swaps;
- enabled and disabled convergence producing identical final selected-region
  bits;
- width one agreeing bitwise with standalone Jacobi;
- every lane agreeing with its independent bounded Dijkstra oracle under the
  established exact/four-ULP fixture policy;
- reached and miss masks computed by the device finalizer after convergence;
- maximum-round and controller/device errors producing no miss bit; and
- all returned controller/status records passing the fixed ABI validators.

Persistent evidence must additionally show one cooperative convergence launch
with no controller copy or host synchronization inside the loop. Record the
actual batch-kernel occupancy and legal grid. Never reuse the Phase 6 probe's
160-block result.

Any failure blocks every later stage. Do not time or profile a partial width,
control, convergence, or load-strategy matrix.

## 3. Real batch-plan and workspace gate

After the real artifact exists, rebuild deterministic Phase 13 features and
the 32/16/8 plan family from it. Validate once-only query assignment and the
exact graph/query/plan fingerprints before uploading any batch. Width one is a
separate singleton baseline; it is not added to the standard planner family.

For every production Jacobi batch:

- deep-validate terminal offsets, union tiles, selected ranges, tile masks,
  and endpoint admission; require the device-materialized representation to
  have no host run-mask or descriptor image;
- use a HIP trace to prove that preparation uploads no graph-wide CSC mask and
  issues no graph-wide mask clear, redundant union-tile upload, initial
  controller H2D, or status memset;
- verify that initialization wave-cooperatively writes the selected
  destination-column runs, including zero masks, before relaxation;
- compare calculated workspace component bytes with retained device capacity;
- record allocation-growth events and reuse across warmed batches;
- sample `hipMemGetInfo` only after resident-graph upload and before each
  workspace-width reservation;
- reject a width when resident graph, explicit reserve, workspace capacity,
  or requested concurrent workspaces exceed observed free memory; and
- record actual selected reset and source-seed traffic separately from total
  allocated distance capacity; and
- verify physical host-control accounting: `N` ordinary polls produce `N + 1`
  controller D2Hs and `N + 1` stream synchronizations in full/evidence mode,
  with terminal transfer completion and lease retirement sharing one boundary.

Do not reinterpret the Phase 13 illustrative 64/8/4-GiB arithmetic as a
runtime memory observation. Do not use compact vertex storage or compact run
descriptors in the Phase 14 HIP engine unless a later separately reviewed
implementation adds and validates those paths.

## 4. Separate correctness campaign on real queries

Select a deterministic, fingerprinted manifest containing at least:

- one-source and multi-source queries;
- complete and padded batches at every standard width;
- low and high union-tile inflation;
- low and high active-lane run masks;
- short, median, and long convergence tails; and
- reachable and bounded-region-miss lanes.

Run every retained batch through both convergence settings and every legal
control mode using `compiler_uniform`. Correctness runs download both distance
slots outside timing and compare every selected lane/vertex with its own
bounded Dijkstra result. Require final label equivalence across controls and
convergence settings, valid reached/miss masks, and full valid-lane slot parity
after convergence.

Run `explicit_wave_broadcast` through the same correctness matrix before it is
eligible for timing. A single label, status, round, or counter-semantics
mismatch rejects that strategy.

## 5. Counter-only work characterization

Use a separate Light or Debug execution, never the final timing samples.
Record per batch and aggregate distributions for:

- considered, visited, skipped, and nonzero CSC runs;
- shared CSC edge records loaded;
- admitted lane-edge relaxations;
- active lanes summed across nonzero runs;
- active destination/lane evaluations and successful decreases;
- executed lane rounds and per-lane convergence/tail rounds;
- lane rounds and lane-edge relaxations avoided by early convergence;
- valid-lane, padded-query-lane, and unused-wave-lane capacity;
- union/selected tile-lane positions and exact union inflation;
- selected two-slot reset and source-seed bytes; and
- requested edge-record and source-distance bytes.

Derive edge reuse as admitted lane-edge pairs divided by shared edge records,
and average active lanes per nonzero run from its exact numerator. Keep these
logical/request counters distinct from profiler-observed memory traffic.
Padded semantic work must remain exactly zero.

## 6. Timing protocol

Only batches that passed the complete correctness gate are eligible. Use
instrumentation None and status-only result transfer. Perform at least five
warmups and thirty retained repetitions per configuration, with a deterministic
interleaving seed that alternates widths, controls, convergence settings, and
load strategies. A single strategy must not receive all warm-cache positions.

Record adjacent intervals for:

1. preparation and metadata transfer;
2. initialization through terminal status on the device timeline;
3. status/result transfer; and
4. end-to-end host wall time, including host polling where applicable.

In host-poll modes, the device-event span includes stream-idle gaps while the
host observes and re-enqueues work; it is not summed kernel-active time. Use a
trace for kernel-active duration. Full distance downloads belong only to the
correctness stage and must not occur inside retained timing samples.

Report P50/P95/P99 batch wall/device-timeline time, per-query latency with its
definition, valid queries per second, and long-tail batch/QueryIds. Compare:

- width one versus 8/16/32;
- per-lane convergence enabled versus disabled;
- persistent versus per-round and every retained chunk size; and
- compiler-uniform versus explicit broadcast.

Batching improves throughput only if the complete correctness-gated,
workload-matched wall-time distribution supports that statement. Otherwise
report no improvement or an inconclusive result.

## 7. Uniform-load versus explicit-broadcast decision

The compiler-uniform path remains the default unless the explicit-broadcast
path passes correctness and shows a repeatable benefit under the identical
real batch manifest. Compare at minimum:

- ordinary and persistent kernels separately;
- all supported widths;
- kernel registers per thread;
- actual active blocks per WGP and cooperative grid size;
- kernel-active and end-to-end distributions;
- L2/read traffic and load-instruction counts from compatible profiler groups;
  and
- effect on long-tail batches and overall query throughput.

Do not retain explicit broadcast because source inspection appears favorable.
If results are mixed, noisy, workload-specific, or reduce legal residency,
keep `compiler_uniform` as the default and record the broadcast experiment as
unselected.

## 8. Trace and profiler runs

Profiler runs use additional processes and the exact correctness-approved
configuration identity. First retain tool and counter availability:

```bash
mkdir -p build/phase14-hip/profiler
rocprofv3 --version \
  >build/phase14-hip/profiler/rocprofv3-version.txt 2>&1
rocprofv3-avail -d 0 list --pmc \
  >build/phase14-hip/profiler/pmc-list.txt 2>&1
rocprofv3-avail -d 0 info --pmc \
  >build/phase14-hip/profiler/pmc-info.txt 2>&1
```

Use the exact trace/PMC command form already validated for the target in
`docs/PHASE12_GPU_VALIDATION.md`. No standalone Phase 14 benchmark driver
exists yet. The HIP CTest `bfnew.batched_jacobi` is the deferred device-
correctness surface only; it is not a throughput, interleaved timing, or
profiler campaign harness. Before measuring throughput or deciding between
compiler-uniform loads and explicit broadcast, add reviewed campaign automation
or invoke Phase 14 through a future combined driver with frozen manifest and
configuration arguments. Until then, do not invent a command or mark profiler
cells measured.

Collect compatible PMC groups in separate processes. At minimum retain the
actual kernel resource record, occupancy/residency, L2/read traffic, memory-unit
activity, and load/instruction data needed for the uniform/broadcast decision.
Unavailable counters remain unavailable; they are never encoded as zero.
Profiler execution is excluded from ordinary timing distributions.

## 9. Required final report

The Phase 14 GPU report must answer, with direct evidence references:

1. Do all lanes, controls, widths, and convergence settings remain correct?
2. How much physical edge reuse becomes logical lane-edge work?
3. What are lane utilization, padding, skipped-run, and union-inflation
   distributions?
4. How much tail work does per-lane convergence avoid, and what does it cost?
5. What is actual selected reset traffic and retained workspace capacity?
6. Does explicit broadcast beat compiler-uniform loads without harming
   register pressure or occupancy?
7. Which width/control/convergence settings maximize total query throughput?
8. Does batching improve throughput over the width-one standalone baseline?

Every conclusion names its graph/query/plan fingerprint, manifest, hardware,
software, configuration, timing stage, and profiler evidence. CPU estimates do
not answer any performance question.

## Current conclusion

All Phase 14 GPU questions are pending. `compiler_uniform` remains the
unmeasured default, `explicit_wave_broadcast` remains an unselected experiment,
and no batch width, control mode, convergence setting, occupancy choice,
latency, throughput, or speedup is recommended. Phase 15 implementation may
proceed independently; this deferred campaign remains reserved until all
implementation phases are complete.
