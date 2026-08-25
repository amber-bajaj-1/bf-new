# Deferred Phase 12 GPU validation and single-query shootout

## Status and execution policy

Do not run these commands during the implementation phases. This file is a
runbook for the combined GPU campaign after all phases are complete and the
user explicitly begins that campaign. It does not authorize browser access,
remote-cloud access by the coding agent, local HIP compilation, or local GPU or
large-graph execution.

Phase 12 currently has bounded CPU and source-level evidence only. The Release
build in `build/phase12-cpu` passed 12/12 bounded CTests in 1.10 seconds, and
the focused ASan/UBSan test passed 1/1 in 0.31 seconds. No HIP compiler or GPU
has executed the shootout driver or any of the three engines in this workspace. No real
1,000-query manifest, device correctness result, actual-kernel occupancy,
warmed timing sample, dependency trace, PMC, long-tail diagnosis, performance
conclusion, or recommended default exists.

All seven questions therefore remain pending measurement:

1. Does persistent cooperative control beat chunked launches?
2. What K minimizes wall time for each ordinary-kernel engine?
3. Does maximum cooperative occupancy help or hurt?
4. Where does dense pull beat sparse frontier push?
5. Is dense chaotic push useful beyond being a diagnostic?
6. Are expensive tails associated with locality, rounds, atomics, or box size?
7. How much old per-round polling time is eliminated?

Recommended defaults are also pending measurement. Every engine, control,
`K`, block-size, and persistent-grid toggle remains configurable.

Run eventually from the `bf-new` project root on the validated target:

- `gfx1151`, wave32;
- 20 runtime-reported HIP multiprocessors/WGPs and 40 architectural CUs;
- cooperative launch support;
- ROCm 7.13.0; and
- ROCprofiler-SDK 1.3.0.

Record the actual values printed by the future run. Do not silently substitute
these expected values if the target reports something else.

The command blocks use Bash arrays and `mapfile`; start a Bash shell before
running them if the target login shell is different.

## Important real-workload prerequisite

The representative input is not currently ready. `out/phase7` contains the
3,958,293,872-byte `xcvu3p.v1.bfgraph`, but
`out/phase7/phase7_partial_report.v1.txt` explicitly records:

```text
all_13_metadata_scan=not_executed_in_interrupted_run
real_query_artifacts=not_emitted_in_interrupted_run
```

The tiny fixtures and a synthetic 1,000-row metadata test are not substitutes
for at least 1,000 representative `logicnets_jscl` queries. The full workload
bridge below imports the device, deeply constructs/validates the global CSR,
CSC, and tile directory, scans all 13 physical netlists, emits the complete
query artifact, and runs its bounded CPU samples. It is intentionally a
separate, potentially very long and global prerequisite. Do not run it locally
under the current policy.

## 1. Release build and tiny correctness gates

Use a fresh build directory. Preserve configure, build, and CTest logs under
the build tree:

```bash
cmake -S . -B build/phase12-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=ON \
  -DBUILD_TESTING=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DBFNEW_FPGA24_DATA_ROOT=/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest \
  -DBFNEW_FPGA24_SCHEMA_ROOT=/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest/fpga-interchange-schema/interchange \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/phase12-hip \
  --target bfnew_device_transfer_test bfnew_jacobi_test \
           bfnew_dense_chaotic_push_gpu_test \
           bfnew_frontier_push_gpu_test bfnew_shootout_test \
           bfnew_shootout_report bfnew_shootout_profile_import \
           bfnew_gpu_shootout \
           bfnew_build_fpga_workload \
  --parallel
ctest --test-dir build/phase12-hip \
  --output-on-failure \
  -R '^(bfnew\.device_transfer|bfnew\.jacobi|bfnew\.dense_chaotic_push_gpu|bfnew\.frontier_push_gpu|bfnew\.shootout)$'
```

The CTest gate uses only tiny fixtures. It must pass before any real workload,
timing, or profiler command. A pass establishes tiny device correctness only;
it does not establish representative correctness or performance.

## 2. Complete the versioned real-workload artifacts

Write new output below `out/phase12-workload` so the partial Phase 7 record is
preserved:

```bash
mkdir -p build/phase12-hip/workload-logs out/phase12-workload
/usr/bin/time -p \
  -o build/phase12-hip/workload-logs/wall.txt \
  build/phase12-hip/bfnew_build_fpga_workload \
    --data-root /Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest \
    --output-root out/phase12-workload \
    --medium vtr_mcml_unrouted.phys \
    --tile-width 8 \
    --tile-height 8 \
    --padding 1 \
  >build/phase12-hip/workload-logs/run.log 2>&1
```

Require all of these outputs and retain their sizes and content fingerprints:

```text
out/phase12-workload/xcvu3p.v1.bfgraph
out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries
out/phase12-workload/vtr_mcml_unrouted.padding1.v1.bfqueries
out/phase12-workload/all_inputs.v1.tsv
out/phase12-workload/phase7_report.v1.txt
```

The bridge already fingerprints the read-only inputs before and after its run.
Require its graph, tile-directory, artifact-determinism, query-validation,
bounded-Dijkstra-sample, and `input_files_unchanged` checks to pass. Preserve
every failure; do not continue with a partial artifact.

## 3. Pilot catalog and representative manifest

The pilot is deliberately two-stage. It computes the four cheap features—
selected-region vertices and edges, fanout, and source count—for the eligible
corpus, then uses a domain-separated prescreen seed to select a stratified
shortlist of at most four times the requested final count. Only that shortlist
runs status-only Jacobi to obtain deterministic completed rounds. The final
selector then applies all five dimensions. This bounds pilot engine work while
preserving a deterministic, recorded selection path; pilot runtime is metadata,
not performance evidence.

Use an explicit selection and ordering seed. The following seed is the Phase 12
reference value, not a hidden default:

```bash
mkdir -p build/phase12-hip/shootout
build/phase12-hip/bfnew_gpu_shootout \
  --stage pilot \
  --workload logicnets_jscl \
  --graph out/phase12-workload/xcvu3p.v1.bfgraph \
  --queries out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries \
  --tile-width 8 \
  --tile-height 8 \
  --selection-seed 20260824 \
  --order-seed 20260824 \
  --warmups 5 \
  --repetitions 30 \
  --query-count 1000 \
  --output build/phase12-hip/shootout
```

Retain `metadata.v1.txt`, `pilot.v1.tsv`, `manifest.v1.tsv`, `catalog.v1.tsv`,
and `configuration-decisions.v1.tsv`. Confirm that `metadata.v1.txt` records
`prescreen_seed = selection_seed ^ 0x7072657363726565`, a
`jacobi_pilot_queries` count no greater than `min(corpus_queries, 4 *
selected_queries)`, and the persisted warmup/timing repetition policy. The
driver must fail if the artifact contains fewer than 1,000 eligible queries, if
query IDs repeat, if the graph/query fingerprints disagree, or if any required
feature is absent. Verify that the manifest contains exactly the requested
number of distinct real query IDs and all five quantile-bin columns.

Create the two bounded built-in adversarial campaigns separately. They do not
accept `--graph`, `--queries`, or `--query-count`, and they never count toward
the real-query minimum:

```bash
for case_name in sparse-wavefront dense-frontier; do
  case_dir="build/phase12-hip/shootout-${case_name}"
  build/phase12-hip/bfnew_gpu_shootout \
    --stage pilot \
    --workload synthetic \
    --case-name "${case_name}" \
    --tile-width 8 \
    --tile-height 8 \
    --order-seed 20260824 \
    --warmups 5 \
    --repetitions 30 \
    --output "${case_dir}"
done
```

Keep each synthetic manifest, catalog, evidence set, and report separate from
the `logicnets_jscl` campaign and from the other synthetic case. Verify the
built-ins exactly before continuing: `sparse-wavefront` has 64 selected
vertices, 63 admitted edges, one source, one target, and target distance 63;
`dense-frontier` has 65 selected vertices, 1,040 admitted edges, one source, 16
targets, and target distance 3 for every target. Synthetic correctness is
bitwise, not four-ULP, for every legal configuration.

## 4. Inspect the automatically resolved legal configuration matrix

For each real engine kernel and block size 128, 256, and 512, query
`hipOccupancyMaxActiveBlocksPerMultiprocessor` with its actual dynamic shared
memory. The matrix contains:

- persistent cooperative with occupancy-derived residency;
- persistent cooperative at every requested fixed blocks-per-WGP value not
  exceeding that kernel's measured limit;
- per-round host polling; and
- chunked host polling at `K = 2,4,8,16,32`.

The pilot creates the full requested matrix internally, resolves it against
the measured limits, writes all acceptance/rejection rows to
`configuration-decisions.v1.tsv`, and persists only accepted tunings in
`catalog.v1.tsv`. Later stages consume that exact catalog; they do not
reconstruct block, chunk, or residency lists from command-line defaults.
Preserve explicit rejection rows for missing occupancy results, illegal block
sizes, and fixed residencies above the kernel limit. Never reuse the Phase 6
probe's 160-block ceiling. Retain
`build/phase12-hip/bfnew_gpu_shootout --help` with the campaign metadata.

## 5. Complete correctness before any measured stage

Run every selected real query and named synthetic case through every legal
configuration. The correctness stage uses instrumentation None, downloads
final labels outside timing, and compares with bounded CPU Dijkstra on the
identical induced subgraph:

```bash
build/phase12-hip/bfnew_gpu_shootout \
  --stage correctness \
  --workload logicnets_jscl \
  --graph out/phase12-workload/xcvu3p.v1.bfgraph \
  --queries out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries \
  --manifest build/phase12-hip/shootout/manifest.v1.tsv \
  --catalog build/phase12-hip/shootout/catalog.v1.tsv \
  --order-seed 20260824 \
  --output build/phase12-hip/shootout
```

Require one passing correctness row for every manifest-query/legal-
configuration pair before continuing. The driver groups correctness execution
by query, assigns ordinals in the actual grouped order, and constructs exactly
one bounded-Dijkstra oracle per query; timing remains interleaved. Oracle
construction copies only the selected tile ranges, traverses only admitted CSR
tile runs, and stores one local label per selected vertex. It must not allocate
or scan a graph-sized CPU distance vector per query. General values use the
documented four-ULP comparison bound; the two built-in synthetic cases require
bitwise equality, and exact fixture coverage is also retained in the tiny
engine tests.
Require stable final bits across repeated controls and separate reached,
bounding-box-miss, maximum-round, overflow, and device/controller error states.
Any missing, failed, duplicate, or fingerprint-mismatched row invalidates the
gate. Do not time a partial matrix.

Repeat correctness for each synthetic directory with `--workload synthetic`,
the matching `--case-name`, manifest, and catalog. Do not pass real artifact
paths to those runs.

## 6. Separate algorithm-counter run

Collect algorithm work outside the timing process. Phase 12 counter evidence
uses Debug so aggregate work and atomic, queue, mask, and overflow details are
available in one separate stage. The driver joins it to correctness/timing by workload
fingerprint, query ID, and stable configuration ID:

```bash
build/phase12-hip/bfnew_gpu_shootout \
  --stage counters \
  --workload logicnets_jscl \
  --graph out/phase12-workload/xcvu3p.v1.bfgraph \
  --queries out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries \
  --manifest build/phase12-hip/shootout/manifest.v1.tsv \
  --catalog build/phase12-hip/shootout/catalog.v1.tsv \
  --correctness build/phase12-hip/shootout/correctness.v1.tsv \
  --instrumentation debug \
  --order-seed 20260824 \
  --output build/phase12-hip/shootout
```

Counter rows have no measured timing and perform no graph-sized distance
download. Debug shared-memory and occupancy requirements can make a timing-
legal configuration Debug-illegal. The counter stage must write
`counter-configuration-decisions.v1.tsv`, run one complete query/configuration
rectangle over only the accepted Debug subset, and leave rejected cells
unavailable; it must not copy a neighboring configuration's counters or encode
an unsupported cell as measured zero. Within accepted rows, preserve rounds,
examined edges, successful decreases, active/frontier work, maximum queue,
atomic attempts and successes, queue claims and duplicate suppressions,
dispatches, host synchronizations, and controller copies. Engine-inapplicable
fields are structural zeros, not evidence that another engine performed no such
work. A zero denominator makes useful-decrease ratio not applicable, not a
measured zero.

## 7. Warmed, interleaved, unprofiled timing

Final timing uses instrumentation None and the common status-only result path.
Upload and retain the immutable graph before warmup. Use the exact same
manifest and legal configuration catalog as correctness, perform five warmup
repetitions, and retain 30 deterministic interleaved repetitions:

```bash
build/phase12-hip/bfnew_gpu_shootout \
  --stage timing \
  --workload logicnets_jscl \
  --graph out/phase12-workload/xcvu3p.v1.bfgraph \
  --queries out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries \
  --manifest build/phase12-hip/shootout/manifest.v1.tsv \
  --catalog build/phase12-hip/shootout/catalog.v1.tsv \
  --correctness build/phase12-hip/shootout/correctness.v1.tsv \
  --warmups 5 \
  --repetitions 30 \
  --order-seed 20260824 \
  --output build/phase12-hip/shootout
```

The driver must reject timing when correctness is incomplete, instrumentation
is requested on a timing command, input/configuration fingerprints differ, or
a catalog configuration is absent. Instrumentation None and status-only result
transfer are enforced internally rather than exposed as timing toggles. Any
retained-workspace allocation growth during a timing sample is also fatal.
The order seed, warmup count, and timing repetition count are persisted in the
manifest. A non-pilot command may omit those options and replay the manifest,
but any explicitly supplied mismatch must fail before graph loading; a wrong
seed must never reinterpret a stable profile-case ordinal.
Preserve preparation GPU time,
SSSP device-timeline time, result-transfer GPU time, and end-to-end wall time
separately for every raw sample. Host-poll device timelines include idle polling
gaps and are not summed kernel-active time.

Repeat the counter and timing commands for each synthetic manifest with its
matching workload identity. Produce separate reports; never combine their
samples or percentiles with the real workload.

## 8. Dependency trace in a separate process

First generate a preliminary report and stable profile-case plan from the
completed correctness, timing, and counter records. This report remains
explicitly pending profiler evidence:

```bash
build/phase12-hip/bfnew_shootout_report \
  --manifest build/phase12-hip/shootout/manifest.v1.tsv \
  --catalog build/phase12-hip/shootout/catalog.v1.tsv \
  --correctness build/phase12-hip/shootout/correctness.v1.tsv \
  --timing build/phase12-hip/shootout/timing.v1.tsv \
  --counters build/phase12-hip/shootout/counters.v1.tsv \
  --profile-plan build/phase12-hip/shootout/profile-plan.v1.tsv \
  --order-seed 20260824 \
  --output build/phase12-hip/shootout/pre-profile-report.v1.json
```

The plan is globally capped at 48 distinct replays. It first selects the worst
available case needed to cover every tail-metric/engine/control stratum, then
fills remaining slots by descending tail severity. Each row repeats the plan
cap, selection reason, covered metrics, workload/input fingerprint, selection
seed, manifest and replay order seeds, warmup/timing policy, schedule policy,
query/configuration identity, and tuning. Reject duplicate `profile_case_id`
values. `--order-seed` defaults to the manifest seed, and an explicit mismatch
must fail.

Do not trace the unprofiled timing process. One profiler process may replay the
whole capped plan by repeating `--profile-case-id` in plan order. Give that
process one unique, nonzero profiler pass ID from the campaign ledger:

```bash
mapfile -t case_ids < <(tail -n +2 \
  build/phase12-hip/shootout/profile-plan.v1.tsv | cut -f1)
profile_case_args=()
for case_id in "${case_ids[@]}"; do
  profile_case_args+=(--profile-case-id "${case_id}")
done
pass_id=REPLACE_WITH_NONZERO_PASS_ID
case_dir="build/phase12-hip/shootout-trace/batch-pass-${pass_id}"
mkdir -p "${case_dir}"
rocprofv3 --version \
  >"${case_dir}/rocprofv3-version.txt" 2>&1
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-directory "${case_dir}" \
  --output-format csv \
  -- build/phase12-hip/bfnew_gpu_shootout \
       --stage profile-case \
       --workload logicnets_jscl \
       --graph out/phase12-workload/xcvu3p.v1.bfgraph \
       --queries out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries \
       --manifest build/phase12-hip/shootout/manifest.v1.tsv \
       --catalog build/phase12-hip/shootout/catalog.v1.tsv \
       --correctness build/phase12-hip/shootout/correctness.v1.tsv \
       --profile-kind trace \
       "${profile_case_args[@]}" \
       --profiler-pass-id "${pass_id}" \
       --order-seed 20260824 \
       --output "${case_dir}" \
  >"${case_dir}/run.log" 2>&1
```

For every case, the process executes exactly one unrecorded, status-only warmup
before launching
`bfnew_shootout_profile_range_begin_marker_kernel`, the measured status-only
replay, and `bfnew_shootout_profile_range_end_marker_kernel`. It writes one
staging row per case plus `profile-range-ledger.v1.tsv` in CLI order. Use that
ledger to pair markers with case/query/configuration IDs. For each case, extract
only kernels strictly between its begin and end marker: exclude graph upload,
the warmup, both marker kernels, other cases, and later output work. Sum those
SSSP/control/finalization kernel durations for `gpu_active_ms` after converting
the raw trace unit to milliseconds.

Inspect ordinary ranges for initialization, ordered round/advance pairs, poll
copies only at the selected control boundary, and final status. Inspect
persistent ranges for exactly one cooperative query kernel and no host
convergence poll. Confirm one graph upload precedes all warmups/replays and that
selected-region distance transfers occur in correctness only.

## 9. PMC compatibility and collection

First preserve the exact device catalog. The `-d 0` selector is global and
precedes the `rocprofv3-avail` subcommand:

```bash
mkdir -p build/phase12-hip/shootout-pmc-check
rocprofv3-avail -d 0 list --pmc \
  >build/phase12-hip/shootout-pmc-check/list.txt 2>&1
rocprofv3-avail -d 0 info --pmc \
  >build/phase12-hip/shootout-pmc-check/info.txt 2>&1
tools/run_phase6_profiler.sh \
  build/phase12-hip \
  build/phase12-hip/shootout-pmc-gate
```

The Phase 6 gate recursively splits incompatible candidate groups. For each
accepted group, launch one fresh batched `profile-case` process over the same
capped case-ID list and retain its nonempty counter CSV and range ledger. Use
`--pmc "${accepted_group[@]}"` exactly as accepted by `pmc-check`; never combine
incompatible groups or use profiler-instrumented runtime as ordinary timing.
Relevant fields include L2 hit/read/write behavior, measured occupancy, memory-
unit activity, waves, and vector/scalar/memory instruction counts. Algorithmic
atomic attempts, queue claims, duplicate suppressions, and barrier cost come
from explicit counters or traces, not an invented hardware proxy.

For a PMC replay, also pass `--profile-kind pmc`, its nonzero
`--profiler-pass-id`, and the nonzero `--profiler-counter-set-id` assigned to
that compatible pass. Do not reuse one pass ID for different counter-set IDs.
If this is a new shell, recreate `profile_case_args` from the profile plan using
the commands in the trace section.

```bash
read -r -a accepted_group <<<"REPLACE_WITH_ONE_ACCEPTED_COUNTER_GROUP"
pmc_pass_id=REPLACE_WITH_NONZERO_PASS_ID
counter_set_id=REPLACE_WITH_NONZERO_COUNTER_SET_ID
pmc_dir="build/phase12-hip/shootout-pmc/set-${counter_set_id}-pass-${pmc_pass_id}"
mkdir -p "${pmc_dir}"
rocprofv3 \
  --pmc "${accepted_group[@]}" \
  --output-directory "${pmc_dir}" \
  --output-format csv \
  -- build/phase12-hip/bfnew_gpu_shootout \
       --stage profile-case \
       --workload logicnets_jscl \
       --graph out/phase12-workload/xcvu3p.v1.bfgraph \
       --queries out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries \
       --manifest build/phase12-hip/shootout/manifest.v1.tsv \
       --catalog build/phase12-hip/shootout/catalog.v1.tsv \
       --correctness build/phase12-hip/shootout/correctness.v1.tsv \
       --profile-kind pmc \
       "${profile_case_args[@]}" \
       --profiler-pass-id "${pmc_pass_id}" \
       --profiler-counter-set-id "${counter_set_id}" \
       --order-seed 20260824 \
       --output "${pmc_dir}" \
  >"${pmc_dir}/run.log" 2>&1
```

Each `profile-case` process writes a validated provenance-only
`trace-case.v1.tsv` or `pmc-case.v1.tsv` with one row per requested case. Its
profiler values are intentionally `unavailable`: the process cannot consume
ROCprof output before the wrapper exits. Review the raw ROCprof CSV, its units,
the begin/end marker ranges, kernel filters, and compatibility group, then
import normalized values with a one-to-one `--case-id` group for every staging
execution ordinal. For a trace batch, run:

```bash
build/phase12-hip/bfnew_shootout_profile_import \
  --kind trace \
  --manifest build/phase12-hip/shootout/manifest.v1.tsv \
  --catalog build/phase12-hip/shootout/catalog.v1.tsv \
  --input "${case_dir}/trace-case.v1.tsv" \
  --case-id REPLACE_WITH_FIRST_CASE_ID \
  --metric gpu_active_ms=REPLACE_WITH_FIRST_RANGE_MILLISECONDS \
  --case-id REPLACE_WITH_SECOND_CASE_ID \
  --metric gpu_active_ms=REPLACE_WITH_SECOND_RANGE_MILLISECONDS \
  --output "${case_dir}/trace-case.measured.v1.tsv"
```

Continue one group per staging row; the example shows two only for brevity. For
a PMC pass, import at least one measured hardware value per case and only values
from that staging file's compatible counter-set ID:

```bash
build/phase12-hip/bfnew_shootout_profile_import \
  --kind pmc \
  --manifest build/phase12-hip/shootout/manifest.v1.tsv \
  --catalog build/phase12-hip/shootout/catalog.v1.tsv \
  --input "${pmc_dir}/pmc-case.v1.tsv" \
  --case-id REPLACE_WITH_FIRST_CASE_ID \
  --metric l2_hit_percent=REPLACE_WITH_NORMALIZED_PERCENT \
  --metric occupancy_percent=REPLACE_WITH_NORMALIZED_PERCENT \
  --case-id REPLACE_WITH_SECOND_CASE_ID \
  --metric l2_hit_percent=REPLACE_WITH_NORMALIZED_PERCENT \
  --metric occupancy_percent=REPLACE_WITH_NORMALIZED_PERCENT \
  --output "${pmc_dir}/pmc-case.measured.v1.tsv"
```

Run `bfnew_shootout_profile_import --help` for every accepted metric name. A
value may be `n/a` or `unavailable`; never turn a missing counter into measured
zero. A multi-row import requires exactly one unique case-ID metric group for
every staging row and rejects missing, extra, duplicate, or shifted case IDs.
It requires measured GPU-active time for every trace row or at least one
measured hardware counter for every PMC row, then reserializes a validator-
approved TSV.

Normalize only after reading the exact counter catalog units and semantics:

- convert trace durations to milliseconds from the raw declared unit;
- convert traffic to bytes using the catalog's unit multiplier—do not assume
  `FETCH_SIZE` or `WRITE_SIZE` is already bytes;
- preserve a counter already reported as percent; convert a documented
  `[0,1]` ratio to percent by multiplying by 100 exactly once;
- when deriving L2 hit percent from raw counts, use
  `100 * sum(hits) / (sum(hits) + sum(misses))` inside one marker range; a zero
  denominator is `n/a`, and unweighted averaging of per-kernel ratios is not a
  substitute;
- sum additive waves/instruction/byte counters only over the measured kernels
  inside that case's marker range; and
- never combine values from different compatibility passes into one imported
  counter-set row.

Raw ROCprof extraction, unit conversion, and ratio derivation remain an
explicit human/compatibility-gated step; only the one-to-one schema import is
automated. Preserve every raw, staging, ledger, and measured file. Combine
measured files under exactly one original header only when their row identities
and compatible counter-set provenance remain distinct.

After collection, update `docs/PROFILER_COUNTERS.md` with the exact accepted and
rejected groups, target/runtime, manifest fingerprint, replayed case IDs, CSV
and normalized-TSV paths, row counts, and failures.

## 10. Deterministic report and evidence review

The report never manufactures prose conclusions. Without `--conclusions`, all
seven answers and the default remain pending even when profiler rows exist.
After reviewing the complete campaign, create
`conclusions.v1.tsv` with this exact header:

```text
schema\trow_type\trecord_id\tstate\tworkload_kind\tworkload_case_id\tworkload_case_name\tconfiguration_ids\tevidence_refs\ttoggles_remain_configurable\ttext
```

Use exactly one `question` row for each record ID 1 through 7 and at least one
positive-ID `recommendation` row. State is `measured` or
`insufficient_evidence`; configuration IDs are comma-separated. Evidence is a
semicolon-separated list of
`kind:execution_ordinal:profiler_pass_id:profiler_counter_set_id:query_id:`
`configuration_id`, where kind is `correctness`, `timing`,
`algorithm_counters`, `trace`, or `pmc`. Non-profiler
references use pass/set `0:0`, trace uses a nonzero pass and set zero, and PMC
uses nonzero pass and set. Every reference must appear exactly in the report's
`supplied_evidence` inventory, and every row must cite at least one known
configuration and evidence record. A measured row must include ordinary timing
plus algorithm-counter, trace, or PMC evidence. Use `n/a` in the toggle column
for questions and `1` for every recommendation; there is no format that can
disable a toggle. Text is one nonempty tab/newline-free field. For example:

```text
1\tquestion\t1\tmeasured\tlogicnets_jscl\tREPLACE_CASE_ID\tlogicnets_jscl\tREPLACE_CONFIG_IDS\ttiming:REPLACE_ORDINAL:0:0:REPLACE_QUERY:REPLACE_CONFIG;algorithm_counters:REPLACE_ORDINAL:0:0:REPLACE_QUERY:REPLACE_CONFIG\tn/a\tREPLACE_WITH_EVIDENCE_BACKED_ANSWER
1\trecommendation\t1\tmeasured\tlogicnets_jscl\tREPLACE_CASE_ID\tlogicnets_jscl\tREPLACE_CONFIG_IDS\ttiming:REPLACE_ORDINAL:0:0:REPLACE_QUERY:REPLACE_CONFIG;pmc:REPLACE_ORDINAL:REPLACE_PASS:REPLACE_SET:REPLACE_QUERY:REPLACE_CONFIG\t1\tREPLACE_WITH_CONFIGURABLE_DEFAULT_RATIONALE
```

The example is not a complete file and contains no evidence claim. Do not use
it until all seven real rows can be populated from measured artifacts.

Generate the final report only from matching, complete artifacts:

```bash
build/phase12-hip/bfnew_shootout_report \
  --manifest build/phase12-hip/shootout/manifest.v1.tsv \
  --catalog build/phase12-hip/shootout/catalog.v1.tsv \
  --correctness build/phase12-hip/shootout/correctness.v1.tsv \
  --timing build/phase12-hip/shootout/timing.v1.tsv \
  --counters build/phase12-hip/shootout/counters.v1.tsv \
  --trace build/phase12-hip/shootout/trace.v1.tsv \
  --pmc build/phase12-hip/shootout/pmc-counter-set-1.v1.tsv \
  --pmc build/phase12-hip/shootout/pmc-counter-set-2.v1.tsv \
  --conclusions build/phase12-hip/shootout/conclusions.v1.tsv \
  --order-seed 20260824 \
  --output build/phase12-hip/shootout/report.v1.json
```

Omit `--trace` or any `--pmc` path only when that evidence was not collected;
the corresponding fields then remain unavailable and no conclusion may depend
on them. Add one `--pmc` argument per compatible normalized counter-set file.
Omit `--conclusions` when the evidence review is incomplete; pending answers
are the only valid output in that state. Do not overwrite the immutable pre-
profile plan when producing the final report.

The report must fail closed on incomplete correctness, duplicate or missing
samples, mixed run kinds, instrumentation/timing violations, mismatched input
fingerprints, or absent legal configurations. Preserve the raw files used to
produce it and reproduce the manifest, catalog, sample TSVs, profile plan, and JSON byte-for-byte from
the same inputs.

Before answering any performance question, verify that the report contains:

- at least 1,000 distinct representative real queries and the separate named
  synthetic cases;
- the five stratification features and bins plus selection/order seeds;
- identical query and graph fingerprints across every compared configuration;
- runtime legality/rejection rows and actual grid/occupancy values;
- complete bounded-Dijkstra correctness before all measured samples;
- P50/P95/P99 wall and SSSP device-timeline distributions;
- throughput with the summed end-to-end wall boundary stated;
- rounds, edges, useful-decrease ratio, frontier/active work, atomic/queue work,
  dispatches, synchronizations, and controller copies;
- separate compatible L2, occupancy, memory-unit, wave, and instruction PMC
  evidence;
- P99/top-tail query IDs with all five workload features; and
- a workload and counter citation for every one of the seven conclusions.

No one counter establishes a bottleneck. Do not merge synthetic and real
percentiles, use unbounded Dijkstra as the correctness oracle, call a
host-poll device timeline kernel-active time, or convert a missing profiler
metric into zero. Recommended defaults may be recorded only after this complete
measured evidence exists, and they must remain configurable.

## Evidence required before changing the current status

Record the exact configure/build/CTest outcomes, tool `--help` and version
output, target/runtime properties, input file fingerprints, artifact-generation
log, manifest and configuration catalogs, correctness rows, raw timing and
counter records, trace and PMC paths, report, every failure/retry, and proof
that both read-only input repositories remained unchanged.

Even a complete Phase 12 campaign establishes controlled standalone
single-query evidence only. It does not implement or validate overlapping
batches, batched expansion, reconstruction integration, congestion updates, or
an adaptive hybrid.
