# Deferred Phase 11 GPU validation

## Status and scope

Do not run these commands during the implementation phases. They are a runbook
for the combined GPU campaign after all implementation phases are complete and
the user explicitly begins that campaign. Phase 11 acceptance is 11/11 bounded
CPU CTests in `build/phase11-cpu-release` (0.09 seconds), plus source-level
checks. The frontier HIP source and HIP test have not been compiled by a HIP
compiler or executed on a GPU, so no device-correctness, occupancy, trace,
profiler, timing, bounded-real, or
performance claim exists.

Run eventually from the `bf-new` project root on the validated AUP target:

- architecture `gfx1151`, wave32;
- 20 runtime-reported HIP multiprocessors/WGPs and 40 architectural CUs;
- cooperative launch support; and
- ROCm 7.13.0 with ROCprofiler-SDK 1.3.0.

Use only the tiny fixture matrix first. Do not substitute the full Phase 7
graph. Bounded-real and full-graph runs remain separate, later steps of the
combined campaign. Phase 12 now supplies the standalone shootout evidence
interface, but neither its implementation nor its acceptance authorizes these
GPU runs automatically. See `docs/PHASE12_GPU_VALIDATION.md` for the later
cross-engine sequence.

## Release build and tiny correctness matrix

Use a fresh build directory and preserve the configure, build, and CTest logs:

```bash
cmake -S . -B build/phase11-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase11-hip \
  --target bfnew_device_transfer_test bfnew_frontier_push_test \
           bfnew_frontier_push_gpu_test \
  --parallel
ctest --test-dir build/phase11-hip \
  --output-on-failure \
  -R '^(bfnew\.device_transfer|bfnew\.frontier_push|bfnew\.frontier_push_gpu)$'
```

The configured executable/CTest names are `bfnew_frontier_push_gpu_test` and
`bfnew.frontier_push_gpu`; `bfnew_frontier_push_test` / `bfnew.frontier_push`
is the portable and structural companion. Do not treat a pass of that CPU
companion as execution of the HIP test.

The HIP matrix uses 17 small fixtures: the 13 Jacobi cases, two dense
contention/shape cases, and expanding-grid and repeated-improvement frontier
cases. It exercises per-round polling, chunked polling at
`K = 2,4,8,16,32`, and persistent cooperative control. It must compare every
converged result with bounded CPU Dijkstra, retain exact
bits across controls, preserve queue-slot parity, and distinguish reached,
bounding-box miss, maximum-round exhaustion, and overflow. The portable
companion separately covers malformed controller state and invalid input. None,
Light, and Debug instrumentation must produce identical labels while
retaining their documented counter separation. The explicit capacity-one
overflow seam must stop without an out-of-bounds queue write in all controls.

The core cooperative case first requests the occupancy-derived grid from the
real persistent frontier kernel and its actual dynamic shared-memory size. It
then requests fixed one- and two-blocks-per-WGP grids only if the runtime
occupancy makes them legal. On the known 20-WGP target these fixed grids are
expected to contain 20 and 40 blocks, but legality and the multiplier must be
runtime-derived. Never reuse the Phase 6 trivial probe's 160-block ceiling.

## Separate unprofiled process runs

Keep ordinary executions separate from every trace and PMC run:

```bash
mkdir -p build/phase11-hip/phase11-unprofiled
for run in 01 02 03 04 05; do
  /usr/bin/time -p \
    -o "build/phase11-hip/phase11-unprofiled/wall-${run}.txt" \
    build/phase11-hip/bfnew_frontier_push_gpu_test \
    >"build/phase11-hip/phase11-unprofiled/run-${run}.log" 2>&1
done
```

These logs measure the complete validation process, including many tiny
queries, setup, and transfers. They are not isolated per-engine, per-control,
or per-grid samples and must not be used for P50/P95/P99, throughput, grid
selection, or an engine recommendation. Isolated unprofiled samples belong to
the eventual combined campaign after all implementation phases, using the
Phase 12 benchmark interface and its separate correctness gate.

`gpu_milliseconds` is a HIP-event device-timeline span from initialization
through terminal status materialization. In host-poll modes, it includes
stream-idle gaps while the host observes the controller and enqueues the next
chunk. It is not summed kernel-active time and is not a pure-GPU basis for
cross-control comparison. Per-kernel active durations require the separately
deferred profiler trace.

## Dependency trace

Capture the complete tiny matrix in a process separate from ordinary timing
and PMC collection:

```bash
mkdir -p build/phase11-hip/phase11-trace
rocprofv3 --version \
  >build/phase11-hip/phase11-trace/rocprofv3-version.txt 2>&1
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-directory build/phase11-hip/phase11-trace \
  --output-format csv \
  -- build/phase11-hip/bfnew_frontier_push_gpu_test \
  >build/phase11-hip/phase11-trace.log 2>&1
```

Inspect the trace for:

- exactly one cooperative frontier query kernel for each persistent run,
  containing initialization, queue rounds/swaps, and terminal status, with no
  host convergence or queue-size observation inside it;
- one initializer followed by ordered frontier-round/controller-advance pairs
  and one finalizer for each ordinary run;
- no controller copy or host synchronization between pairs within a chunk;
- one initial controller H2D, one controller D2H per ordinary poll, and one
  terminal controller D2H;
- occupancy-derived plus runtime-legal fixed one- and two-blocks-per-WGP
  cooperative grids in the core case; and
- one resident graph upload per fixture/engine construction followed by reuse
  across that fixture's control matrix.

For `N` ordinary polls, the result must report `N + 2` physical controller
copies and `N + 1` host synchronizations. The copies are initial workspace
H2D, `N` poll observations, and terminal validation D2H. The synchronizations
are the polls plus one shared final-result/same-stream-retirement boundary.
Timing-event elapsed queries add no synchronization. Persistent has
`N = 0` and no convergence poll. The portable model intentionally counts
semantic host observations instead, so its values are not physical-copy
expectations.

The whole-matrix trace is structural evidence. It includes setup/transfers and
cannot establish latency or throughput. An asynchronous copy name is not by
itself proof of overlap when the host endpoint is pageable.

## PMC compatibility and counter collection

First preserve the device catalog and independently check candidate groups.
The `-d 0` selector is global and precedes the `rocprofv3-avail` subcommand:

```bash
mkdir -p build/phase11-hip/phase11-pmc-check
rocprofv3-avail -d 0 list --pmc \
  >build/phase11-hip/phase11-pmc-check/list.txt 2>&1
rocprofv3-avail -d 0 info --pmc \
  >build/phase11-hip/phase11-pmc-check/info.txt 2>&1
rocprofv3-avail -d 0 pmc-check \
  GRBM_COUNT GRBM_GUI_ACTIVE GPUBusy OccupancyPercent \
  MeanOccupancyPerCU MeanOccupancyPerActiveCU MemUnitBusy \
  >build/phase11-hip/phase11-pmc-check/utilization.txt 2>&1 || true
rocprofv3-avail -d 0 pmc-check \
  SQ_WAVES_sum SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_INSTS_VALU \
  SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_FLAT SQ_INSTS_TEX_LOAD \
  SQ_INSTS_TEX_STORE \
  >build/phase11-hip/phase11-pmc-check/instruction-wave.txt 2>&1 || true
rocprofv3-avail -d 0 pmc-check \
  FETCH_SIZE WRITE_SIZE L2CacheHit GL2C_HIT_sum GL2C_MISS_sum \
  GL2C_MC_RDREQ_sum GL2C_MC_WRREQ_sum GL2C_EA_WRREQ_64B_sum \
  GL2C_WRREQ_STALL_max WriteUnitStalled \
  >build/phase11-hip/phase11-pmc-check/l2-memory.txt 2>&1 || true
```

Use the existing Phase 6 gate to recursively split rejected combinations and
record device-compatible passes on its non-SSSP workload:

```bash
cmake --build build/phase11-hip \
  --target bfnew_gpu_probe bfnew_gpu_barrier_benchmark \
  --parallel
tools/run_phase6_profiler.sh \
  build/phase11-hip \
  build/phase11-hip/phase11-pmc-gate
```

Then collect every accepted pass on the frontier matrix in its own profiler
process. Run this snippet with Bash because it uses tab-delimited records and
arrays:

```bash
mkdir -p build/phase11-hip/phase11-frontier-pmc
pass=0
while IFS=$'\t' read -r label counters; do
  pass=$((pass + 1))
  read -r -a pmcs <<<"${counters}"
  pass_directory="build/phase11-hip/phase11-frontier-pmc/pass-${pass}-${label}"
  mkdir -p "${pass_directory}"
  rocprofv3 \
    --pmc "${pmcs[@]}" \
    --output-directory "${pass_directory}" \
    --output-format csv \
    -- build/phase11-hip/bfnew_frontier_push_gpu_test \
    >"${pass_directory}/run.log" 2>&1
done <build/phase11-hip/phase11-pmc-gate/accepted-pmc-groups.txt
```

Require and retain a nonempty counter-collection CSV for every accepted pass.
The tiny test internally asserts invariants for admitted edges, decreases,
active frontier vertices, queue claims, duplicate suppressions, and atomic
attempts/successes, but it does not serialize successful-run counter values.
Do not claim numeric algorithm-counter evidence from its logs. Export and
analysis of those values uses the Phase 12 benchmark/reporting interface during
the deferred campaign. Use compatible PMC passes only for their collected memory-traffic, wave,
occupancy, and utilization fields. No one PMC proves an atomic or queue
bottleneck; do not combine incompatible passes or present profiler-instrumented
time as ordinary runtime.

The HIP ABI reports aggregate `active_vertices` and `active_lane_rounds`, plus
maximum/empty/small-frontier counters. It does not export the portable model's
exact `current_frontier_sizes` sequence. Neither the dependency trace nor PMC
collection should be described as a per-round queue-size trace unless the
Phase 12 driver explicitly records one.

## Evidence required before any GPU claim

Record the exact target/runtime versions, HIP compile outcome, complete tiny
CTest result, actual frontier-kernel occupancy, legal grids executed, trace
paths, accepted PMC groups and CSVs, and every failure. Even a clean tiny
matrix remains only tiny-fixture device-correctness and structural evidence.
It does not validate bounded-real or full-graph behavior and does not establish
a performance recommendation. Isolated per-mode timing, kernel-duration
analysis, bounded-real/full-graph cases, and the three-engine conclusion all
remain pending for the eventual combined GPU campaign.
