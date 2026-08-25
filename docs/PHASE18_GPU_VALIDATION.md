# Phase 18 Deferred GPU and End-to-End Validation

## Status and acceptance boundary

Phase 18 local acceptance is bounded and HIP-off. The implemented portable
path compacts terminal query results, reconstructs requested target paths from
final distance labels and incoming CSC, preserves stable-edge-ID ordering and
zero-weight-cycle backtracking, and validates exact path-label tightness,
reported cost, endpoint continuity, source/target termination, and final
selected-region membership. The optional HIP path and the end-to-end campaign
surface are source implementations awaiting the device procedure below.

The settled HIP source surface is
`include/bfnew/hip/compact_path_results.hpp`,
`src/hip/compact_path_results.hip.cpp`, and
`tests/compact_path_results_test.hip.cpp`. All three batched engines expose
`run_compact_paths(...)`; the Phase 17 executor adds the explicit
`compact_paths` transfer mode, and the existing artifact campaign selects it
with `--transfer paths`.

Final local evidence is HIP-off Release `19/19` in `2.02` seconds, including
`bfnew.compact_paths` `1/1` in `0.10` seconds; ASan+UBSan `19/19` in `2.82`
seconds, including `bfnew.compact_paths` `1/1` in `0.16` seconds. Strict host
source/public-header syntax and fake-`__HIPCC__` production, CLI, and deferred-
test syntax passed warning-clean. The focused CPU executable/CTest pair is
`bfnew_compact_paths_test` / `bfnew.compact_paths`. The optional HIP pair is
`bfnew_compact_path_results_test` / `bfnew.compact_path_results`, and the
HIP+FPGAIF campaign executable is `bfnew_gpu_batched_expansion`.

No browser, cloud service, AUP host, HIP compiler, GPU, large/full graph, or
real benchmark query artifact was used during local acceptance. Consequently
device correctness, compact-copy traffic, GPU-event timing, occupancy,
physical memory traffic, all-query throughput, latency distributions,
path-quality distributions, and an evidence-based production configuration
remain unavailable, not zero.

The required `logicnets_jscl.padding1.v1.bfqueries` artifact is still absent.
A `.phys` and `.netlist` pair may be used only through the deterministic Phase
7 artifact-generation workflow; it is not a direct substitute for the
versioned `.bfqueries` input expected by the campaign.

## 1. Preserve source, artifact, and device identity

Run from the project root. Record the exact source revision or source-tree
fingerprint, build configuration, graph/query content fingerprints, compiler,
HIP runtime, operating system, and target device before accepting evidence.

```bash
mkdir -p build/phase18-campaign/evidence
uname -a >build/phase18-campaign/evidence/uname.txt
cmake --version >build/phase18-campaign/evidence/cmake-version.txt
hipcc --version >build/phase18-campaign/evidence/hipcc-version.txt
rocminfo >build/phase18-campaign/evidence/rocminfo.txt
```

The expected device target is `gfx1151` with wave size 32 and cooperative
launch support. Stop and record any mismatch. Never reuse a Phase 6 probe
limit as the occupancy of a Phase 14--16 engine or the Phase 18 reconstruction
kernel.

Fingerprint both versioned artifacts before the run and retain the importer
configuration that generated them:

```bash
phase18_graph=out/phase12-workload/xcvu3p.v1.bfgraph
phase18_queries=out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries
test -f "$phase18_graph"
test -f "$phase18_queries"
shasum -a 256 "$phase18_graph" "$phase18_queries" \
  >build/phase18-campaign/evidence/input-sha256.txt
```

The graph/query loader must deeply validate the graph, spatial directory,
tile runs, device layout, every query, and their shared identity before graph
upload or timing begins.

## 2. Build the optional HIP path

Use a clean HIP-enabled build for the bounded device gate:

```bash
cmake -S . -B build/phase18-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase18-hip --parallel \
  --target bfnew_compact_path_results_test
```

For the artifact-to-result campaign, enable the independently reviewed FPGA
Interchange bridge and point it at the read-only schema root:

```bash
phase18_schema_root=/absolute/path/to/fpga-interchange-schema/interchange
test -d "$phase18_schema_root"
cmake -S . -B build/phase18-campaign \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=ON \
  -DBFNEW_FPGA24_SCHEMA_ROOT="$phase18_schema_root" \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase18-campaign --parallel \
  --target bfnew_gpu_batched_expansion
build/phase18-campaign/bfnew_gpu_batched_expansion --help \
  >build/phase18-campaign/evidence/campaign-help.txt
```

Configuration must fail clearly when HIP or an explicitly requested FPGAIF
dependency is unavailable. Successful compilation is source evidence, not
device correctness.

## 3. Run the bounded device correctness gate

Run only the dedicated Phase 18 device test first:

```bash
ctest --test-dir build/phase18-hip \
  -R '^bfnew[.]compact_path_results$' \
  --output-on-failure
```

The bounded gate must cover:

- Jacobi pull, dense chaotic push, and active-frontier push final label
  layouts, including both Jacobi distance slots;
- widths 1, 8, 16, and 32, valid padded lanes, and canonical target order;
- per-target distance, reachability, selected-source validity, edge-count
  length, and explicit reconstruction status;
- target-is-source zero-edge paths and canonical multi-source selection;
- equal-cost and parallel alternatives ordered by stable logical `EdgeId`;
- a zero-weight cycle and a lower-ID tight dead end that require real
  path-local backtracking;
- fractional and unequal positive weights, long cross-tile adjacency, and
  spill-tile membership;
- later-generation success after expansion, full-region fallback success,
  clean unreachable targets, and explicit engine/identity failures, including
  rejection of all-complete and reconstruction-failure miss payloads before
  retry;
- deterministic vertices, stable edge IDs, compact path-label values, and
  results across repetitions; and
- exact agreement with Dijkstra on each final admitted induced region.

Every sampled path must pass two independent checks. The ordinary compact
validator checks its compact per-path labels, source and target endpoints,
simple vertex sequence, stable-edge continuity, exact tightness, ordinary-
float accumulated cost, selected source, and selected tiles. A bounded
diagnostic replay may additionally compare against the complete final lane
image. That diagnostic copy is correctness evidence only and must not enter a
timed production pass.

## 4. Prove the compact transfer boundary

The production result path must never download an entire lane-distance matrix.
Trace every D2H operation and account for its bytes. The permitted payload is
limited to fixed-size per-target summaries plus path-sized arrays containing
the reconstructed vertices, stable edge IDs, and compact path-label values.
Any status/controller observation algorithmically required by an ordinary
host-poll control mode is reported separately from the final compact result
payload. The compact payload/status/error subtotal excludes controller polls.
Per-round and chunked controls report `controller_poll_count` and
`controller_poll_bytes`; persistent reports both as zero. The checked overall
D2H total is the compact subtotal plus controller-poll bytes, and poll bytes
must equal `controller_poll_count * sizeof(DeviceController)` with the frozen
96-byte controller ABI.

For each target, record:

```text
target VertexId
distance
reached status
selected-source validity and VertexId
edge-count path length
reconstruction status
path offset/counts
```

For a complete path, vertices and compact labels have `path_length + 1`
entries and stable edge IDs have `path_length` entries. A source target has
one vertex/label and zero edges. An unreachable or query-failure summary has
no invented source, path, or finite distance. A finite label for which the
reconstructor cannot reach a canonical source is `no_tight_path` and fails
the campaign; it is never published as reached success.

The resident device image and transferred vertex/edge arenas use checked
32-bit words. Charge four D2H bytes per path vertex and stable edge ID; widening
an edge ID into the host's strong 64-bit type does not double transfer traffic.

Every valid lane in a clean batch produces one identity-bound summary payload;
padding produces none. Compact paths are published only from a terminal
generation. Before a query is admitted to retry, its clean miss payload must
contain at least one unreachable target and no status other than complete or
unreachable. An all-complete miss or a miss with any reconstruction failure is
an explicit engine failure and must not expand. An admitted retry discards its
prior complete/unreachable summaries and paths together with the prior label,
queue, mask, controller, and convergence state. A terminal full-region miss
retains its per-target complete/unreachable classification. QueryId, terminal
arrays, and terminal maps remain unchanged across generations. A reached
status with any noncomplete target is an explicit engine failure, never
success.

Use HIP traces to prove the boundary. After defining `phase18_command` with the
frozen helper in Section 5, create a distinct trace-only output identity and
run exactly that command under the profiler:

```bash
phase18_trace_output="$phase18_output_root/trace.$phase18_schedule.tsv"
set_phase18_command \
  jacobi persistent "$phase18_schedule" "$phase18_trace_output"
test ! -e "$phase18_trace_output"
test ! -e "$phase18_trace_output.tmp"
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-format csv \
  -- "${phase18_command[@]}"
```

Retain raw trace files and a normalized byte ledger. Reject a production run
that contains a `vertex_count * lane_width * sizeof(float)` D2H label copy.
The portable `final_result_serialization` fields are sizing/model evidence only
and omit compact summaries discarded by retries. Physical claims must use the
separate retry-inclusive HIP compact payload/status/error subtotal, controller-
poll count/bytes, and overall D2H fields and reconcile all of them with this
trace.

## 5. Execute the complete no-congestion pipeline

The accepted pipeline order is:

```text
load preprocessed graph/query artifacts
upload graph exactly once
plan all initial query batches
run exactly one selected SSSP engine/control configuration
collect misses, expand, replan, and restart as required
reconstruct final compact target paths
transfer and validate compact results
```

The frozen driver requires explicit graph, query, output, engine, schedule,
control, `overlap-greedy` planner, transfer mode, maximum expansion count, and
terminal policy. `--transfer paths` additionally requires positive warmup,
repetition, and quality-sample counts plus an explicit quality seed. Path mode
also requires `--reconstruction backtracking` and `--instrumentation none`;
there is no hidden reconstruction or hybrid-engine selector. Preserve the
captured help text and exact invocation with every report. Lane width accepts
1, 8, 16, or 32; width 1 is the scalar baseline. The example remains width 32.

This Bash helper expands one fully explicit command. The shown `one-ring`
schedule and seed are reproducible campaign identities, not recommendations.
Repeat the entire matched matrix for each expansion schedule still under
consideration; the helper supplies the required schedule-specific argument.

```bash
phase18_exe=build/phase18-campaign/bfnew_gpu_batched_expansion
phase18_output_root=build/phase18-campaign/evidence/paths
phase18_schedule=one-ring
mkdir -p "$phase18_output_root"

set_phase18_command() {
  local engine="$1"
  local control="$2"
  local schedule="$3"
  local output="$4"
  local -a schedule_arguments=()
  case "$schedule" in
    one-ring|doubling)
      ;;
    fixed-ring)
      schedule_arguments=(--fixed-ring-size 2)
      ;;
    hybrid)
      schedule_arguments=(--hybrid-small-expansions 2)
      ;;
    *)
      return 2
      ;;
  esac

  phase18_command=(
    "$phase18_exe"
    --graph "$phase18_graph"
    --queries "$phase18_queries"
    --output "$output"
    --engine "$engine"
    --schedule "$schedule"
    "${schedule_arguments[@]}"
    --control "$control"
    --planner overlap-greedy
    --tile-width 8
    --tile-height 8
    --lane-width 32
    --maximum-expansions 4
    --terminal-policy fallback
    --rounds-per-chunk 8
    --block-size 256
    --grid occupancy
    --maximum-rounds 0
    --per-lane-convergence on
    --transfer paths
    --reconstruction backtracking
    --instrumentation none
    --warmups 5
    --repetitions 30
    --quality-sample-count 100
    --quality-seed 1729
  )
}

for engine in jacobi dense frontier; do
  for control in persistent chunked per-round; do
    phase18_output="${phase18_output_root}/${engine}."
    phase18_output+="${control}.${phase18_schedule}.tsv"
    test ! -e "$phase18_output"
    test ! -e "$phase18_output.tmp"
    set_phase18_command \
      "$engine" "$control" "$phase18_schedule" "$phase18_output"
    "${phase18_command[@]}"
  done
done
```

The first warmup invocation is the separately reported cold, capacity-growing
execution. Remaining warmups and every measured repetition reuse the resident
graph, engine, reconstruction workspace, and grown capacities in the same
process. The report records each repetition's enclosing host duration,
applicable SSSP/reconstruction/result-transfer device spans, compact payload/
status/error subtotal, controller-poll count/bytes, and overall D2H bytes;
nearest-rank host P50/P95/P99; retry-inclusive process transfer totals; final
`NoCongestionResultAccounting`; and the quality sample's selected QueryIds,
per-target observations, and aggregate metrics.

Run the `logicnets_jscl` artifact first. Run a representative, fingerprinted
subset through every one of these nine engine/control pairs:

```text
jacobi  x persistent, chunked, per-round
dense   x persistent, chunked, per-round
frontier x persistent, chunked, per-round
```

Use the same query subset, planner, batch width, expansion policy, correctness
gate, warmups, repetitions, and recorded deterministic process order for all
pairs. Each process executes one configuration's in-process cold run, warmups,
and repetitions; the driver does not claim cross-configuration interleaving.
If temporal drift must be controlled, rotate or randomize the recorded order
of whole fresh-output processes in an external campaign layer. Control-
specific settings such as chunk size or cooperative grid remain part of
configuration identity. Do not compare unlike correctness or transfer modes.

The recommended configuration may run over all available benchmark queries
only after its engine, control, batch width, planner thresholds, and expansion
schedule are selected from complete comparable evidence. If Phase 12 or Phase
17 evidence still supplies no unique recommendation, stop and report that
fact rather than choosing by enum order or intuition. Remaining contest
artifacts may run only after `logicnets_jscl` completes and as resource limits
permit.

No step adds congestion, resource ownership, routing disjointness, occupancy
updates, historical costs, path-based early stopping, or an exit-edge
certificate.

## 6. Timing protocol and stage ledger

Report both host wall time and GPU-event time where the latter is meaningful.
CPU-only stages have no invented GPU duration. Use checked accumulation and
require the declared stage ledger to cover the complete claimed boundary.
For `--transfer paths`, the cold execution, every additional warmup, and every
measured repetition must each provide measured SSSP, reconstruction, and
result-transfer GPU-event evidence. Reject the entire campaign if any such
execution lacks it; do not substitute zero or omit the run.

Report these stages separately:

1. cold artifact load and deep validation;
2. one resident graph upload;
3. initial and retry batch planning;
4. SSSP execution;
5. geometric selected-region growth only;
6. controller/orchestration, including validation, batch-description
   preparation, controller polling, failed-lane/retry collection, restart
   bookkeeping, and terminal-ledger assembly not already included in SSSP;
7. GPU reconstruction and compact-path packing;
8. compact result transfer;
9. `warm_all_query`, measured with the graph resident and reusable capacities
   already grown: it is the checked exact sum of stages 3--8 when every named
   host interval is measured, or a distinct enclosing host observation when
   asynchronous HIP leaves one or more named host stages unpartitionable and
   therefore unavailable;
10. `cold_execution`, a separate enclosing observation of the first all-query
    execution after upload, including workspace construction or capacity
    growth when they occur; and
11. `cold_pipeline`, the checked exact sum of stage 1, stage 2, and that
    distinct `cold_execution`, from artifact open through validated compact
    results.

Record warmup count, measured repetition count, process order, per-run raw
durations, P50/P95/P99, total queries/second, path targets/second, and result
bytes. The detailed warm stages and `warm_all_query` belong to a pre-grown
measured pass. A named host stage that cannot be partitioned in an asynchronous
HIP pass remains unavailable; it is not recorded as measured zero and is not
folded into controller/orchestration. `cold_execution` belongs to the separate
first capacity-growing pass, so it is neither replaced by nor required to
equal `warm_all_query`. The cold pipeline is not a warm throughput sample.
Profiler and counter runs are separate from ordinary timing. A host-poll
device-event span includes stream-idle host dependency gaps and is not summed
kernel-active time.

Stage accounting must identify whether preparation or capacity growth belongs
to the reported interval. Do not subtract overlapping event spans, add host
and device clocks, or relabel Phase 17 cold-controller timing as Phase 18 warm
end-to-end evidence.

## 7. Path-quality sample

Select the quality sample before observing results. The implemented method
orders candidates by SplitMix64 of the recorded seed and QueryId, with QueryId
as the collision tie-break, so input ordering cannot change the sample. Record
corpus fingerprint, sampling frame, seed, sample size, exclusions, and the
number of targets reachable in both the final bounded region and the unbounded
graph.
Include every complete target in the selected results, even when another target
made that query terminal-unreachable in the full region. Define zero bounded
over zero unbounded cost/length as unit relative inflation. Define a positive
bounded value over a zero unbounded baseline as positive infinity while
retaining its finite absolute inflation; do not drop or abort that observation.
For each sampled target, compare the validated bounded compact path with
unbounded Dijkstra using the same frozen weights.

Report at least:

- bounded and unbounded cost;
- absolute and relative cost inflation;
- bounded and unbounded edge count;
- absolute and relative path-length inflation; and
- P50/P95/P99 plus maximum for both inflation measures.

The bounded correctness oracle remains Dijkstra on the final selected induced
subgraph. A larger cost or path length than unbounded Dijkstra is a quality
observation, not a correctness failure. Never expand a reached query merely
to improve this metric.

## 8. Required report and fail-closed rules

Every accepted report binds:

- source, graph, query, device-layout, and execution-configuration identities;
- hardware, OS, compiler, HIP runtime, and profiler versions;
- engine, control, chunk/grid, planner, lane width, convergence, expansion,
  fallback, and reconstruction settings;
- canonical query count and a complete partition into reached compact results
  or explicit terminal dispositions;
- canonical target count and reconstruction-status counts;
- target-summary, vertex, edge-ID, path-label, and compact-payload subtotal
  bytes;
- fixed compact status/error traffic separately from public compact arenas;
- controller-poll count/bytes separately from the compact subtotal, with zero
  polls for persistent control and a checked overall D2H sum;
- path edge-count and selected-region distributions;
- initial success, retry, fallback, unreachable, and failure metrics;
- every timing stage and its clock domain; and
- quality-sample definition and results.

Reject duplicate or missing QueryIds, generation mismatches, changed terminal
maps, target-count mismatches, incomplete reached queries, complete paths on a
failed query, all-complete misses, reconstruction-failure misses admitted to
retry, invalid selected sources, offset/count overflow, malformed path shapes,
non-simple paths, non-tight compact labels, edge discontinuity, selected-region
escape, incorrect cost, unknown statuses, missing GPU-event evidence from any
cold/warmup/repetition execution, and any aggregate overflow. Refuse an
existing final output or stale temporary output, and publish a report atomically
only after all ledgers validate.

## Current conclusion

The Phase 18 bounded portable implementation and deferred HIP/campaign source
surface are not device-performance evidence. No representative artifact or
GPU run exists locally, so no engine/control combination, expansion schedule,
latency, throughput, compact-transfer rate, or path-inflation distribution is
reported. The full-region fallback remains a completion policy, not a global-
optimality certificate.

Phase 18 stops before congestion/resource-conflict logic and adaptive hybrid
selection. The completed Phase 19 audit is
`docs/PHASE19_FINAL_AUDIT.md`; it adds no algorithm and records the current
insufficient-evidence/no-recommendation result without relabeling this deferred
runbook as executed evidence.
