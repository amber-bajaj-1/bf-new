# Deferred Phase 9 GPU validation

## Status and scope

Do not run these commands during the CPU-only implementation phases. They are
instructions for the eventual combined GPU campaign after all implementation
phases are complete. Phase 9 passed locally only through its portable CPU
semantic/control tests and a text-level HIP source audit. The HIP library and
`bfnew.jacobi` test have not been compiled or executed, and no occupancy,
profiler, counter, or GPU timing result is recorded here.

Run from the `bf-new` project root on the validated AUP target:

- architecture `gfx1151`, wave32;
- 20 runtime-reported HIP multiprocessors/WGPs and 40 architectural CUs;
- cooperative launch supported; and
- ROCm 7.13.0 with ROCprofiler-SDK 1.3.0.

The Phase 9 test matrix is intentionally tiny. Do not substitute the full
Phase 7 graph. Bounded-real and full-graph validation remain deferred to the
combined GPU campaign.

## Build and correctness tests

Use a new Release build so the evidence is not mixed with local CPU output:

```bash
cmake -S . -B build/phase9-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase9-hip \
  --target bfnew_device_transfer_test bfnew_jacobi_test \
  --parallel
ctest --test-dir build/phase9-hip \
  --output-on-failure \
  -R '^(bfnew\.device_transfer|bfnew\.jacobi)$'
```

The configured executable and CTest names are `bfnew_jacobi_test` and
`bfnew.jacobi`. The test runs the 13 tiny HIP cases through per-round polling,
chunked polling at `K = 2,4,8,16,32`, and persistent cooperative control. The
Phase 5 core case additionally runs occupancy-derived persistent control and
explicit one- and two-blocks-per-WGP grids. It obtains the legal ceiling from
the real persistent Jacobi kernel and rejects either explicit request if it is
not legal. On a 20-WGP device, those dynamically checked fixed launches are
expected to be 20 and 40 blocks; the test does not use the Phase 6 probe's
160-block ceiling.

Preserve the configure, build, and CTest logs. A passing run must also preserve
bitwise results for representable cases, four-ULP agreement for the fixed
general-weight cases, identical results among controls, converged two-column
identity, explicit final-slot parity, reached/miss semantics, and zero distance
atomic counters. It must also reject the same-shaped resident-graph fixture
whose one changed CSC weight produces a different deterministic two-word
fingerprint. That fingerprint catches accidental identity mismatches; it is not
a cryptographic validation mechanism.

The maximum-round fixture must pass independently in persistent, per-round,
and `K=32` chunked control. The chunked result must retain all 32 already-
submitted round/advance pairs while counting only the single round that
actually executes; the remaining pairs are device-done no-ops.

## Separate unprofiled validation runs

Profiler-instrumented time is not ordinary runtime. First record several
unprofiled executions in their own directory:

```bash
mkdir -p build/phase9-hip/phase9-unprofiled
for run in 01 02 03 04 05; do
  /usr/bin/time -p \
    -o "build/phase9-hip/phase9-unprofiled/wall-${run}.txt" \
    build/phase9-hip/bfnew_jacobi_test \
    >"build/phase9-hip/phase9-unprofiled/run-${run}.log" 2>&1
done
```

This command measures one complete validation-process wall time, not an
isolated engine/control configuration. The test internally validates that its
per-query GPU-event and host-wall fields are nonnegative, but it does not
serialize those fields. Do not derive per-mode P50/P95/P99, throughput, or a
default grid from these logs. Isolated benchmark code belongs to the Phase 12
shootout interface now implemented at the host evidence layer; running its
device driver and the complete performance campaign still waits until the user
begins the combined GPU campaign. See `docs/PHASE12_GPU_VALIDATION.md`.

`gpu_milliseconds` is the elapsed HIP event device-timeline span from
initialization through status materialization. In host-poll modes that interval
includes stream-idle gaps while the CPU reads the controller and re-enqueues
work. It is not summed kernel-active time and must not be compared across
control modes as pure GPU work. Per-kernel durations require the deferred
profiler evidence.

The current validation binary also cannot isolate ordinary round-dispatch cost
or the persistent kernel's grid-barrier contribution from its relaxation work.
The Phase 6 barrier microbenchmark is a separate probe kernel and must not be
presented as Jacobi barrier cost. Those measurements need an isolated
Phase 12 benchmark interface during the eventual combined campaign before the
original Phase 9 profiling checklist can be claimed complete.

## Dependency trace

Capture tracing in a separate process from both unprofiled timing and PMC
collection:

```bash
mkdir -p build/phase9-hip/phase9-trace
rocprofv3 --version \
  >build/phase9-hip/phase9-trace/rocprofv3-version.txt 2>&1
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-directory build/phase9-hip/phase9-trace \
  --output-format csv \
  -- build/phase9-hip/bfnew_jacobi_test \
  >build/phase9-hip/phase9-trace.log 2>&1
```

This complete-matrix trace is structural validation only. Inspect it for:

- one cooperative Jacobi query kernel for each persistent run, with no
  convergence-state copy inside that launch;
- one initialization followed by ordered round/advance pairs for ordinary
  runs;
- one initial controller H2D, one poll D2H and synchronization per per-round
  check or completed chunk, and one terminal controller D2H, never a poll or
  synchronization between pairs inside a chunk;
- explicit occupancy-derived, 20-block, and 40-block persistent launches in
  the core case; and
- one graph upload per fixture/engine construction, followed by reuse across
  that fixture's control-mode matrix.

The trace covers many tiny test cases and includes setup and result transfers.
It is not latency or throughput evidence, and an asynchronous copy name alone
does not prove host/device overlap for a pageable endpoint.

For `N` ordinary convergence polls, the HIP work record counts `N + 2`
physical controller copies: initial H2D, `N` poll D2H records, and terminal
D2H. Persistent has zero convergence polls but still counts its initial and
terminal controller transfers. Host synchronizations are `N + 1`: the polls
plus one shared final result-transfer/same-stream-retirement boundary.
The timing events are already complete at that terminal synchronization
and add no separate synchronization. These physical counts deliberately differ
from the portable model, whose controller-copy field represents semantic host
control observations.

## PMC compatibility and counter collection

The device selector for `rocprofv3-avail` is global and must precede its
subcommand. Preserve the catalog and compatibility output:

```bash
mkdir -p build/phase9-hip/phase9-pmc-check
rocprofv3-avail -d 0 list --pmc \
  >build/phase9-hip/phase9-pmc-check/list.txt 2>&1
rocprofv3-avail -d 0 info --pmc \
  >build/phase9-hip/phase9-pmc-check/info.txt 2>&1
rocprofv3-avail -d 0 pmc-check \
  GRBM_COUNT GRBM_GUI_ACTIVE GPUBusy OccupancyPercent \
  MeanOccupancyPerCU MeanOccupancyPerActiveCU MemUnitBusy \
  >build/phase9-hip/phase9-pmc-check/utilization.txt 2>&1 || true
rocprofv3-avail -d 0 pmc-check \
  SQ_WAVES_sum SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_INSTS_VALU \
  SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_FLAT SQ_INSTS_TEX_LOAD \
  SQ_INSTS_TEX_STORE \
  >build/phase9-hip/phase9-pmc-check/instruction-wave.txt 2>&1 || true
rocprofv3-avail -d 0 pmc-check \
  FETCH_SIZE WRITE_SIZE L2CacheHit GL2C_HIT_sum GL2C_MISS_sum \
  GL2C_MC_RDREQ_sum GL2C_MC_WRREQ_sum GL2C_EA_WRREQ_64B_sum \
  GL2C_WRREQ_STALL_max WriteUnitStalled \
  >build/phase9-hip/phase9-pmc-check/l2-memory.txt 2>&1 || true
```

A rejected combined candidate group is expected and must be split before
collection. The existing Phase 6 gate script performs that split recursively,
records every accepted group, and verifies one nonempty collection on its
non-SSSP workload:

```bash
cmake --build build/phase9-hip \
  --target bfnew_gpu_probe bfnew_gpu_barrier_benchmark \
  --parallel
tools/run_phase6_profiler.sh \
  build/phase9-hip \
  build/phase9-hip/phase9-pmc-gate
```

Then collect each accepted pass on the complete tiny Jacobi matrix. Run this
snippet with Bash because it uses tab-delimited records and arrays:

```bash
mkdir -p build/phase9-hip/phase9-jacobi-pmc
pass=0
while IFS=$'\t' read -r label counters; do
  pass=$((pass + 1))
  read -r -a pmcs <<<"${counters}"
  pass_directory="build/phase9-hip/phase9-jacobi-pmc/pass-${pass}-${label}"
  mkdir -p "${pass_directory}"
  rocprofv3 \
    --pmc "${pmcs[@]}" \
    --output-directory "${pass_directory}" \
    --output-format csv \
    -- build/phase9-hip/bfnew_jacobi_test \
    >"${pass_directory}/run.log" 2>&1
done <build/phase9-hip/phase9-pmc-gate/accepted-pmc-groups.txt
```

Require a nonempty counter-collection CSV for every accepted pass and retain
the exact accepted-group file. Use explicit Jacobi statistics for examined CSC
edges and useful decreases; use PMC passes for L2/read traffic, waves,
occupancy, and utilization. No available PMC directly counts algorithmic
changed-flag updates or cooperative barrier cost. Do not infer a bottleneck
from one counter, combine incompatible passes, or report profiler-instrumented
runtime as ordinary timing.

## Evidence still required before a GPU claim

After the eventual run, record the exact target/runtime versions, HIP compile
result, CTest result, real-kernel occupancy ceiling, legal grids actually
executed, trace paths, accepted PMC passes, and any failures. Even a complete
tiny-matrix pass does not validate a bounded-real graph or establish a
performance recommendation. Keep those items, isolated ordinary-dispatch and
Jacobi-barrier measurements, per-mode timing, and full campaign conclusions
explicitly pending until the combined GPU campaign.
