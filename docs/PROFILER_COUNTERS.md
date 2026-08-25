# Phase 6 Profiler Counters

## Status

Phase 6 passed under the maintainer's CPU-only acceptance policy. No current
GPU counter run was requested or collected, and no profiler result is claimed.
The commands below are optional instructions for a maintainer with access to an
AUP `gfx1151` target.

Expected target contract:

- ROCm 7.13.0;
- ROCprofiler-SDK 1.3.0;
- `gfx1151`, wave32;
- 20 HIP multiprocessors/WGPs and 40 physical CUs; and
- PC sampling not exposed.

## Required commands

The device selector is global and precedes the subcommand:

```bash
rocprofv3-avail -d 0 list --pmc
rocprofv3-avail -d 0 info --pmc
rocprofv3-avail -d 0 pmc-check <counter names>
```

The Phase 6 script begins with the required utilization, instruction/wave, and
L2/memory candidate groups and recursively splits every rejected group. It
retains the raw check output and writes the final groups to
`build-hip/phase6-profile/accepted-pmc-groups.txt`.

## Accepted PMC groups

Not collected. If the optional target run is performed, record the exact
accepted groups and run date here and list rejected/unavailable single counters
rather than silently omitting them.

## Optional actual collection

Not collected. The optional evidence is a nonempty `counter_collection.csv`
from `bfnew_gpu_probe --workload-only`, which launches the repeated
arithmetic/memory kernel 64 times over 1,048,576 float elements. If run, record
the selected counter pass, CSV path, row count, and byte size here.

Profiler-instrumented runtime is not production timing and must not be reported
as an SSSP result. The Phase 6 workload is not SSSP.

## Phase 12 shootout evidence status

Not collected. Phase 12 implements explicit profiler-evidence fields and keeps
`unavailable`, `not_applicable`, and `measured` distinct, but it does not invent
accepted PMC groups or numeric values. No engine/control/block/K/grid
configuration has a current L2, occupancy, memory-unit, wave, or instruction
measurement.

When the combined GPU campaign is explicitly started, follow
`docs/PHASE12_GPU_VALIDATION.md`. Rerun the compatibility gate on the exact
target/runtime, retain one raw output and nonempty CSV per accepted group, and
use `bfnew_shootout_profile_import` only after human review of raw counter
names, units, kernel filters, and compatibility. The importer binds normalized
values one-to-one to stable execution ordinals in a provenance-only
`profile-case` staging batch and revalidates its manifest, catalog, run kind,
pass, and counter-set identity. Marker ledgers delimit each measured range from
its in-process warmup and neighboring cases. Record here:

- run date, GPU architecture, HIP and ROCprofiler-SDK versions;
- runtime-reported WGP count and actual per-engine/per-block occupancy limits;
- every accepted counter group exactly as passed to `rocprofv3`;
- every rejected or unavailable singleton counter;
- raw compatibility-output and counter-CSV paths plus row counts and sizes;
- staging and normalized profiler-TSV paths and the exact imported values;
- the shootout manifest/input fingerprint and replayed configuration IDs; and
- any failed, partial, retried, or incompatible pass.

PMC collection must use a process separate from correctness, unprofiled timing,
algorithm counters, and dependency tracing. Do not average incompatible passes,
encode a missing metric as measured zero, infer an algorithmic atomic or queue
count from a hardware proxy, or use profiler-instrumented runtime in
P50/P95/P99 or throughput. Until those records are added, the seven Phase 12
questions and every recommended default remain pending measurement.

## Phase 19 final profiler audit

Phase 19 finds no locally accepted SSSP profiler evidence. The fixed final
inventory has ten canonical typed rows: marker-delimited GPU-active time plus
normalized values for:

- L2 hit percentage;
- L2 read bytes;
- L2 write bytes;
- occupancy percentage;
- memory-unit busy percentage;
- waves;
- vector instructions;
- scalar instructions; and
- memory instructions.

The existing compatibility script begins with utilization,
instruction/wave, and L2/memory candidate groups. Those candidate names and the
recursive split/import machinery are implemented evidence infrastructure, not
accepted measurements. No compatible target group, marker ledger, raw
nonempty SSSP trace/PMC file, normalized import, or matching Phase 18
configuration/tail provenance exists locally.

Every Phase 19 profiler field and bottleneck conclusion therefore remains
`unavailable`, with no numeric value or evidence fingerprint. In particular,
the audit does not infer atomic contention from algorithm counters, queue cost
from work counts, memory bottlenecks from analytical bytes, or kernel-active
time from a host-poll GPU-event span. The required zero payload on an
`unavailable` row is an absence sentinel and is never interpreted as measured
zero.

In a future measured report, every profiler row must share the profiler-
bottleneck configuration fingerprint, and that selection must match the
best-engine row. Percentage rows are limited to 0 through 100. Nonfinite,
negative, and signed-negative-zero encodings are rejected so serialized
evidence remains canonical.

The local Phase 19 report has zero performance-profiled SSSP features and no
production recommendation. Its exact evidence disposition and the additional
missing artifact/device/baseline gates are recorded in
`docs/PHASE19_FINAL_AUDIT.md`.
