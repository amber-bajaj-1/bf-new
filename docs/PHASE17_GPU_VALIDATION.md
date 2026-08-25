# Phase 17 Deferred GPU and Real-Corpus Validation

## Status and acceptance boundary

Phase 17 local acceptance is bounded and HIP-off. The implementation supplies
the deterministic all-query orchestrator, all four required expansion
schedules, explicit fallback/failure policy, portable adapters for all three
batched engines, and a HIP-gated compact-status executor for those same
engines. A HIP+FPGAIF-gated `bfnew_gpu_batched_expansion` driver loads the
versioned graph/query artifacts, executes one explicit engine/schedule
candidate over every query, and writes a fail-closed fingerprinted TSV report.
That driver exists in source but has not been HIP-compiled or run. Local tests
use only a tiny synthetic graph. No browser, cloud
service, real query artifact, full graph, HIP compiler, or GPU was used.

Final local evidence is Release `18/18` in `1.70` seconds, including Phase 17
`1/1` in `0.01` seconds, and ASan+UBSan `18/18` in `3.32` seconds, including
Phase 17 `1/1` in `0.19` seconds. The Phase 17 test/fixture and production core
passed strict C++20 warnings. The deferred HIP test passed strict host and
fake-HIP `__HIPCC__` syntax; all seven new or modified batch/expansion HIP
translation units passed the strict fake-HIP check. These are source checks,
not HIP compilation or device execution.

The absent `logicnets_jscl.padding1.v1.bfqueries` artifact prevents the
required first all-query corpus run. Consequently the initial success-rate,
expansion distribution, retry utilization, real replanning cost, repeated
work, final-region size, fallback/unreachable rate, throughput, and expansion-
schedule recommendation are unavailable for the representative workload. A
zero in an unavailable field is not a measurement. Phase 17 deliberately has
no schedule default: `select_expansion_schedule_from_evidence()` returns a
recommendation only from one comparable record for every required schedule,
and only when the best evidence score is unique. A tie returns no schedule.

This procedure is reserved for the combined campaign after implementation
phases, when a maintainer has the query artifact and access to the validated
AUP `gfx1151` system.

## 1. Preserve source, input, and target identity

Run from the project root and retain the exact source revision or source-tree
fingerprint, graph/query fingerprints, query count, compiler/runtime identity,
and device identity.

```bash
mkdir -p build/phase17-hip/evidence
uname -a >build/phase17-hip/evidence/uname.txt
cmake --version >build/phase17-hip/evidence/cmake-version.txt
hipcc --version >build/phase17-hip/evidence/hipcc-version.txt
rocminfo >build/phase17-hip/evidence/rocminfo.txt
```

The expected target has wave size 32, `gfx1151`, 20 HIP
multiprocessors/WGPs, cooperative launch support, ROCm 7.13.0, and
ROCprofiler-SDK 1.3.0. Stop and record any mismatch. Never substitute the
Phase 6 probe ceiling for occupancy of an actual Phase 14–16 engine kernel.

Record a content fingerprint for
`logicnets_jscl.padding1.v1.bfqueries`. Validate the artifact against the
matching graph before execution. A `.phys` and `.netlist` pair is not a
substitute for this exact preprocessed query artifact unless the reviewed
Phase 7 query-generation workflow deterministically regenerates it and records
the resulting fingerprint.

## 2. Build the optional HIP path

Use a clean HIP-enabled build without enabling FPGAIF during the bounded
device gate:

```bash
cmake -S . -B build/phase17-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase17-hip --parallel \
  --target bfnew_batched_expansion_hip_test
```

Configuration must fail clearly if HIP is unavailable. Successful compilation
is source evidence, not device correctness.

The real-corpus driver is gated by both HIP and FPGA Interchange support. In a
separate clean campaign build, set the schema root to the actual read-only
FPGA Interchange schema directory on the target:

```bash
phase17_schema_root=/absolute/path/to/fpga-interchange-schema/interchange
test -d "$phase17_schema_root"
cmake -S . -B build/phase17-campaign \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=ON \
  -DBFNEW_FPGA24_SCHEMA_ROOT="$phase17_schema_root" \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase17-campaign --parallel \
  --target bfnew_gpu_batched_expansion
build/phase17-campaign/bfnew_gpu_batched_expansion --help \
  >build/phase17-campaign/bfnew_gpu_batched_expansion.help.txt
```

The target does not exist unless both gates are enabled. Preserve the captured
`--help` text with the build evidence. The commands below use only options in
that implemented help surface.

## 3. Run the bounded device correctness gate

Run only the dedicated bounded device test first:

```bash
ctest --test-dir build/phase17-hip \
  -R '^bfnew\.batched_expansion_gpu$' \
  --output-on-failure
```

The expected executable/CTest pair is `bfnew_batched_expansion_hip_test` /
`bfnew.batched_expansion_gpu`. The HIP-off pair is
`bfnew_batched_expansion_test` / `bfnew.batched_expansion`. The deferred device
CTest has a configured 300-second timeout.

The bounded device gate must cover:

- Jacobi pull, dense chaotic push, and active-frontier push;
- one geometric ring, a fixed larger ring, doubled x/y margins, and hybrid
  small-first-then-doubling schedules;
- several misses in one batch, exactly one missed lane, retry-local lane
  compaction, and dissimilar expanded regions;
- a long cross-tile edge whose intermediate endpoint is beyond one ring;
- a spill-terminal route admitted only after its located intermediate enters;
- success after several source restarts, one final full-region fallback,
  explicit expansion-limit/stalled-region failure, and a globally unreachable
  full-region query;
- deterministic QueryId order, incremented generation, preserved terminal
  mappings, original-source reseeding, and no retained label/queue leakage;
- clean reached/miss partitioning, maximum-round and engine-error exits that
  expand no lane, and no missed target reported as success; and
- agreement with Dijkstra on every terminal lane's final induced region.

The current deferred `bfnew.batched_expansion_gpu` source exercises the compact
classification/retry workflow but deliberately downloads no distance image.
It therefore cannot by itself satisfy the last Dijkstra bullet. Before the
combined campaign treats the device gate as complete, add a separate bounded
correctness replay or temporary diagnostic that downloads terminal labels from
the underlying engine for the same retry plans and retains the final-region
Dijkstra comparison. Do not weaken compact production transfer to obtain that
evidence.

Run compact transfer with `InstrumentationLevel::none`. It must copy exactly
one 48-byte `DeviceRunStatus` after each batch finalizer and download no
controller, per-lane trace, counter block, or graph-sized distance image.
Persistent control therefore performs one final compact status transfer; the
ordinary controls still perform their algorithmically required intermediate
controller observations. Separately run the evidence transfer mode with the
instrumentation required by its engine and confirm that work evidence is
explicitly `measured` or `unavailable`, never inferred from zero.

## 4. Prove restart and compact-status safety

For every missed lane, retain the retry trace containing original QueryId,
old and new expansion generation, prior/final selected tiles, planning pass,
batch/lane identity, reached/miss masks, and terminal disposition.

Each retry must rebuild the plan and retained run masks from the retry-local
queries. The selected engine must execute its ordinary selected-only reset and
source-seed path. It must not retain or consume labels, frontier queues,
activity masks, changed flags, controller masks, or convergence records from a
prior generation. The original source and target arrays and their terminal
maps remain byte-identical; only selected tiles and expansion generation may
change.

A query whose targets are reached is removed before expansion. Algorithm or
controller errors, maximum-round exhaustion, and queue overflow are terminal
engine failures, not bounding-region misses. A clean converged status must
partition every valid lane into exactly reached or miss before the host
collects failed QueryIds.

Per-query generation, retry/expansion-count, or margin overflow must terminate
only that query as `identity_or_count_overflow` without wrapping or aborting
unrelated queries. Aggregate telemetry, histogram, or campaign-count overflow
is not that disposition: it must abort the campaign fail-closed and leave no
apparently complete evidence report.

## 5. Evaluate all four schedules on identical work

Use identical graph/query fingerprints, query order, batch policy, engine,
control, convergence setting, block/grid policy, expansion limit, terminal
policy, warmups, and repetition order for:

```text
one geometric tile ring
fixed larger ring
doubling x/y margins
hybrid small-first then doubling
```

The fixed ring size and hybrid small-first count are configuration identities
and must be reported. Interleave schedule executions to limit ordering and
thermal bias. A full-region fallback is a separate final restart and occurs at
most once. Region growth must be strict; a schedule that admits no new tile
uses the configured fallback or terminates as `region_stalled`.

Do not select a default from intuition, the bounded synthetic fixture, CPU wall
time, or incomplete schedule evidence. Compare one complete evidence record
per schedule. Every record must bind the same nonzero execution-configuration
fingerprint in addition to the graph, query, engine, planner, control, limit,
and terminal-policy identity; transfer/load or portable-representation changes
must produce a different identity. Rank correct completion first; only when
work is available and comparable may it enter the decision. If the best score
is tied, record that no schedule was selected. Preserve the exact unique
selected record that caused `select_expansion_schedule_from_evidence()` to
return a recommendation.

## 6. Run the real corpus in the required order

Run every `logicnets_jscl` query first. Only after that corpus completes may
the remaining contest benchmarks run as resource limits permit.

The implemented driver requires explicit graph, query, output, engine, and
schedule choices. It refuses to overwrite either the final output or a stale
`.tmp` file. The following Bash matrix executes all three engines and all four
schedules twice: compact/None for the production transfer boundary, then
evidence/Light for comparable work records. It fixes every shared option and
uses the schedule-specific arguments required by `--help`.

This driver is an all-query classification/diagnostic and cold-controller
evidence path, not the final warm timing harness. It synchronizes the resident
graph upload before the controller starts, then creates a cold reusable engine
workspace whose first growth is included in the controller timing. Capacity is
reused only among batches and retries inside that one process.

```bash
set -euo pipefail
phase17_exe=build/phase17-campaign/bfnew_gpu_batched_expansion
phase17_graph=out/phase12-workload/xcvu3p.v1.bfgraph
phase17_queries=out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries
phase17_output=build/phase17-campaign/logicnets-jscl
test -x "$phase17_exe"
test -f "$phase17_graph"
test -f "$phase17_queries"
mkdir -p "$phase17_output"

run_phase17_candidate() {
  local engine="$1"
  local schedule="$2"
  local transfer="$3"
  local instrumentation="$4"
  local output="$5"
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

  "$phase17_exe" \
    --graph "$phase17_graph" \
    --queries "$phase17_queries" \
    --output "$output" \
    --engine "$engine" \
    --schedule "$schedule" \
    "${schedule_arguments[@]}" \
    --tile-width 8 \
    --tile-height 8 \
    --lane-width 32 \
    --maximum-expansions 4 \
    --terminal-policy fallback \
    --control chunked \
    --rounds-per-chunk 8 \
    --block-size 256 \
    --grid occupancy \
    --maximum-rounds 0 \
    --per-lane-convergence on \
    --transfer "$transfer" \
    --instrumentation "$instrumentation"
}

for engine in jacobi dense frontier; do
  for schedule in one-ring fixed-ring doubling hybrid; do
    run_phase17_candidate \
      "$engine" "$schedule" compact none \
      "$phase17_output/$engine.$schedule.compact.v1.tsv"
    run_phase17_candidate \
      "$engine" "$schedule" evidence light \
      "$phase17_output/$engine.$schedule.evidence-light.v1.tsv"
  done
done
```

`--maximum-rounds 0` is the documented request for `V + 1`. An occupancy grid
must not include `--blocks-per-wgp`; a fixed grid must include a positive
value. Compact transfer requires `--instrumentation none`; evidence transfer
requires `light` or `debug`. Use unique output directories for repetitions,
because an existing output is a deliberate hard failure.

Each report begins with schema `bfnew.batched-expansion.v1` and includes file
and device-layout fingerprints, the execution-configuration and schedule-
comparison fingerprints, complete configuration, aggregate metrics, expansion
histogram, canonical per-query outcomes, selected tile IDs, and batch trace
identities. Confirm the exact timing declarations
`resident_graph_upload=completed-before-controller`,
`workspace_initial_state=cold-reused-within-campaign`, and
`timing_boundary=controller-including-first-workspace-growth`. Confirm that the
four schedule candidates within one engine/transfer/configuration have the
same nonzero comparison fingerprint before comparing them. A nonzero exit
caused by engine or per-query identity/count
failure still publishes a diagnostic report and must not be accepted as
successful evidence. Aggregate telemetry overflow instead aborts fail-closed
and must not leave an apparently complete final report. Conversely, exit zero
does not mean every query was reachable; inspect all terminal disposition
counts.

Only after the entire `logicnets_jscl` matrix passes may the remaining emitted
contest-query artifacts run. For the currently documented medium artifact,
set:

```bash
phase17_queries=out/phase12-workload/vtr_mcml_unrouted.padding1.v1.bfqueries
phase17_output=build/phase17-campaign/vtr-mcml-unrouted
mkdir -p "$phase17_output"
```

Then repeat the exact engine/schedule/transfer loops above. Apply the same rule
to any additional reviewed `.bfqueries` artifacts as resource limits permit.

For correctness sampling, request terminal distance images and compare each
lane with Dijkstra on its final admitted region. For final timing, a later
in-process repeated driver/pass must use compact status,
`InstrumentationLevel::none`, a synchronized resident graph, pre-grown warm
workspace capacity, and no graph-sized result download. The current one-pass
CLI cannot supply that warm boundary, so do not relabel its derived throughput
as final timing. Keep correctness, counter, trace, profiler, and final timing
processes separate.

The all-query driver intentionally uses compact or status-and-work execution;
it has no distance-download option and is not the bounded Dijkstra replay.
Likewise, its `initial_planning_nanoseconds`, `replanning_nanoseconds`,
`execution_nanoseconds`, `total_nanoseconds`, and derived milliqueries/second
come from the Phase 17 controller. Graph upload is synchronized before that
boundary, while the first cold workspace growth is inside it. The report does
not independently time graph loading/upload, compact result transfer,
individual queries, or active GPU kernels. Use the separate correctness replay,
in-process warm repetitions, trace, and profiler stages for those missing
boundaries rather than attributing them to the aggregate driver fields.

## 7. Required metrics and report

For every schedule and engine, report:

1. initial reached queries divided by input queries;
2. the complete total-expansion histogram, separating scheduled expansions
   from final full-region fallbacks;
3. failed lanes divided by their originating valid-lane capacity, plus retry
   valid lanes divided by retry lane capacity;
4. initial planning, replanning, device execution, compact result-transfer,
   and end-to-end wall time as separate stages;
5. repeated selected-edge estimate and, when measured, shared edge-record work
   and logical lane-edge work;
6. final selected tile, vertex, and edge distributions;
7. reached, unreachable-in-full-region, expansion-limit, region-stalled, and
   engine-failure counts;
8. batch count, attempts/query, long-tail retries, and fallback count; and
9. total all-query throughput and per-query latency distributions.

Host planning nanoseconds are CPU wall observations, not GPU time. Logical
edge work is not a physical memory transaction. Use trace/profiler evidence
from the selected underlying engine to report actual kernel occupancy, L2 or
memory traffic, and stalls; do not transfer the Phase 6 probe or a different
engine's measurements.

Every conclusion names source/input fingerprints, hardware/software identity,
engine, full schedule parameters, planner/control/convergence configuration,
the execution-configuration and comparison fingerprints, transfer/
instrumentation mode, expansion and terminal policy, timing stage, and
repetition set. A tied top score is reported as no selection, never resolved by
schedule order or an undocumented default.

## Current conclusion

The Phase 17 orchestration, HIP-gated compact-status boundary, and HIP+FPGAIF-
gated all-query report driver exist in source, but no real HIP compiler or GPU
has validated them. The driver has not consumed either real artifact. No real-
corpus success rate, retry distribution, schedule comparison, target
occupancy, physical memory behavior, latency, or throughput is available.
Therefore no expansion schedule is the production default. The full-region
fallback is a correctness completion policy, not an exit lower-bound
certificate, and reached queries are accepted on their admitted induced
regions without expansion. Even after the first driver run exists, its
controller throughput will include cold first-workspace growth and must not be
reported as warm final performance.

Phase 17 validation stops before compact target/path result production and
reconstruction integration. Phase 18 now implements that later boundary under
`docs/PHASE18_GPU_VALIDATION.md`; it does not retroactively change this Phase
17 campaign or add congestion.
