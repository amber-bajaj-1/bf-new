# Deferred Phase 10 GPU validation

## Status

Do not run these commands during the CPU-only implementation phases. They are
instructions for the combined GPU campaign after all phases are complete.
Phase 10 passed locally through bounded CPU semantics/tests and source-level
checks only. Neither a HIP compiler nor a GPU executed the dense engine or its
HIP test, and no device-correctness, occupancy, trace, counter, timing, or
performance result is recorded here.

Run only on the validated AUP `gfx1151` target with ROCm 7.13.0 and
ROCprofiler-SDK 1.3.0. Use the tiny fixture matrix first. Do not substitute the
full Phase 7 graph.

## HIP build and tiny correctness matrix

From the `bf-new` project root:

```bash
cmake -S . -B build/phase10-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase10-hip \
  --target bfnew_device_transfer_test bfnew_jacobi_test \
           bfnew_dense_chaotic_push_gpu_test \
  --parallel
ctest --test-dir build/phase10-hip \
  --output-on-failure \
  -R '^(bfnew\.device_transfer|bfnew\.jacobi|bfnew\.dense_chaotic_push_gpu)$'
```

The Phase 10 executable/CTest names are
`bfnew_dense_chaotic_push_gpu_test` and
`bfnew.dense_chaotic_push_gpu`. Preserve configure, build, and CTest logs.

The dense test repeats the Phase 9 tiny cases and adds high-fan-in and
hub-fan-out cases. Every case runs per-round control, chunked control at
`K = 2,4,8,16,32`, and persistent cooperative control. It compares final
labels with bounded CPU Dijkstra, requires bit equality for exact fixtures and
the established four-ULP policy for seeded general weights, and requires all
controls and repeated runs to finish with identical distance bits.

The core case requests occupancy-derived plus explicit one- and two-blocks per
WGP cooperative grids. Legality must come from the real dense persistent
kernel's occupancy result. Do not use the Phase 6 probe's 160-block ceiling as
a dense limit. The maximum-round case must pass independently in persistent,
per-round, and `K=32` control; the final chunk submits 32 pairs but only one
round executes.

## Dependency trace

Capture trace evidence separately from ordinary execution and PMC collection:

```bash
mkdir -p build/phase10-hip/phase10-trace
rocprofv3 --version \
  >build/phase10-hip/phase10-trace/rocprofv3-version.txt 2>&1
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-directory build/phase10-hip/phase10-trace \
  --output-format csv \
  -- build/phase10-hip/bfnew_dense_chaotic_push_gpu_test \
  >build/phase10-hip/phase10-trace.log 2>&1
```

Inspect the trace for:

- one cooperative dense query kernel per persistent run, containing
  initialization and status with no convergence copy inside it;
- one initializer followed by ordered dense-round/controller-advance pairs
  and one finalizer in ordinary modes;
- no controller copy or host synchronization between pairs inside a chunk;
- one initial controller H2D, one D2H per poll, and one terminal controller
  D2H; and
- one resident graph upload per fixture followed by reuse across its control
  matrix.

For `N` ordinary polls, the result should report `N + 2` physical controller
copies and `N + 1` host synchronizations. The synchronizations are the polls
plus one shared final result-transfer/lease-retirement boundary; timing-
event elapsed queries add none. Persistent has `N = 0`. Queue and frontier
counters must remain zero. Debug atomic attempts must equal admitted edges
examined, successful atomic updates must equal successful decreases, and full-
edge rounds must equal executed rounds.

This trace covers many tiny cases and includes setup/transfers. It is
structural evidence, not latency or throughput evidence. An async copy name is
also not proof of overlap for pageable host memory.

## Optional unprofiled process checks

If the combined campaign records repeat stability, keep it separate from
profiler runs:

```bash
mkdir -p build/phase10-hip/phase10-unprofiled
for run in 01 02 03 04 05; do
  /usr/bin/time -p \
    -o "build/phase10-hip/phase10-unprofiled/wall-${run}.txt" \
    build/phase10-hip/bfnew_dense_chaotic_push_gpu_test \
    >"build/phase10-hip/phase10-unprofiled/run-${run}.log" 2>&1
done
```

These are complete validation-process times, not isolated engine/mode times.
In host-poll modes, the engine's event span includes idle gaps while the host
observes and re-enqueues work; it is not summed kernel-active time. Do not use
these runs for P50/P95/P99, throughput, grid choice, or a Jacobi-versus-dense
recommendation. Isolated benchmarks use the implemented host Phase 12 evidence
interface, but its device campaign is
still deferred; use `docs/PHASE12_GPU_VALIDATION.md`.

## Profiler counters during the combined campaign

First rerun the existing Phase 6 profiler gate to obtain device-compatible PMC
groups:

```bash
cmake --build build/phase10-hip \
  --target bfnew_gpu_probe bfnew_gpu_barrier_benchmark \
  --parallel
tools/run_phase6_profiler.sh \
  build/phase10-hip \
  build/phase10-hip/phase10-pmc-gate
```

Only then collect each accepted group on the dense test in a separate process.
Retain the accepted-group file and every nonempty CSV. Relevant later analysis
includes L2/read/write traffic, waves/occupancy, admitted edges, successful
decreases, atomic attempts/successes, changed-block publications, and
high-contention destinations. No one counter establishes a bottleneck; do not
combine incompatible passes or treat profiler-instrumented time as ordinary
runtime.

## Evidence still pending

Before claiming GPU validation, record the exact target/runtime versions, HIP
compile outcome, complete tiny CTest result, real dense-kernel occupancy and
legal grids, trace paths, accepted PMC passes, and any failures. Bounded-real
and full-graph runs, isolated per-mode timing, and cross-engine conclusions
remain deferred until the combined campaign.
