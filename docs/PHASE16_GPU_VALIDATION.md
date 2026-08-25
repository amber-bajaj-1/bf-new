# Phase 16 Deferred GPU Validation

## Status and execution policy

Phase 16 local acceptance uses bounded HIP-off CPU tests only. Final
post-hardening Release evidence is `17/17` in `1.56` seconds, including Phase
16 `1/1` in `0.02` seconds. ASan+UBSan evidence is `17/17` in `3.03` seconds,
including Phase 16 `1/1` in `0.16` seconds. Do not run this GPU procedure during
implementation phases. It is reserved for the combined campaign after all
implementation phases are complete and a maintainer has access to the
validated AUP `gfx1151` environment.

The deferred HIP test passed strict-warnings host syntax and fake-HIP
`__HIPCC__` syntax, including its device probe. Phase 16 public headers passed
strict host syntax, and both production HIP translation units passed strict
fake-HIP syntax. These source checks did not invoke a HIP compiler or GPU. No
browser, cloud service, or large/full-graph test was used.

No Phase 16 HIP translation unit has been compiled by a HIP compiler or
executed on a GPU locally. Device correctness, queue/mask atomic behavior,
actual occupancy, physical memory traffic, latency, throughput, batching
benefit, and comparison with the other batched engines are unavailable. A
zero in an unavailable hardware field is not a measured zero.

The current first device path uses the provisional full-vertex/retained-CSR
workspace choice. The real `logicnets_jscl.padding1.v1.bfqueries` artifact is
absent, so representative batch plans, union distributions, queue high-water
marks, runtime memory capacity, and real-corpus throughput cannot yet be
measured.

## 1. Record the target and source identity

Run from the project root on the target system and retain the exact source
revision or source-tree fingerprint, compiler, runtime, and device identity.
Do not substitute the Phase 6 probe occupancy ceiling for a real frontier
kernel occupancy result.

```bash
mkdir -p build/phase16-hip/evidence
uname -a >build/phase16-hip/evidence/uname.txt
cmake --version >build/phase16-hip/evidence/cmake-version.txt
hipcc --version >build/phase16-hip/evidence/hipcc-version.txt
rocminfo >build/phase16-hip/evidence/rocminfo.txt
```

The expected campaign target has wave size 32, `gfx1151`, 20 HIP
multiprocessors/WGPs, cooperative launch support, ROCm 7.13.0, and
ROCprofiler-SDK 1.3.0. Stop and record the mismatch if the environment differs;
do not silently reinterpret results as the known target.

## 2. Configure and compile the optional HIP path

Use a clean HIP-enabled build without enabling FPGAIF or attempting a real
corpus:

```bash
cmake -S . -B build/phase16-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase16-hip --parallel \
  --target bfnew_batched_frontier_push_hip_test
```

Configuration must fail clearly when HIP is unavailable. Compilation success
is source evidence only; it is not device correctness.

## 3. Run the bounded device correctness gate

Run only the dedicated bounded Phase 16 CTest first:

```bash
ctest --test-dir build/phase16-hip \
  -R '^bfnew\.batched_frontier_push_gpu$' \
  --output-on-failure
```

The exact executable target is `bfnew_batched_frontier_push_hip_test`; the
CTest name is `bfnew.batched_frontier_push_gpu`, with a configured timeout of
240 seconds. The separate HIP-off executable/CTest pair is
`bfnew_batched_frontier_push_test` / `bfnew.batched_frontier_push`.

The deferred test must cover:

- widths 1, 8, 16, and 32;
- persistent cooperative, per-round host polling, and chunked host polling at
  `K = 2,4,8,16,32`;
- `enable_per_lane_convergence=true|false`;
- low-prefix validity and padded lanes;
- independent, multi-source, and shared-source lanes;
- lane frontiers that empty immediately, after one round, and after several
  rounds while another lane continues;
- bounded unreachable targets;
- shared-destination lane merging, high fan-in, and repeated improvements to
  a vertex already represented in the next queue;
- zero-weight edges/cycles and strict-decrease termination;
- maximum-round, initialization-overflow, round-overflow, invalid-controller,
  and malformed-input exits;
- canonical strictly increasing source/target slices, unique valid-lane query
  IDs, and empty/canonical padded-lane terminals and metadata;
- None/Light/Debug instrumentation without any final-distance change;
- repeated use of one retained workspace lease; and
- status-only and distance-download entry points.

The full distance-download result must normalize every padded-lane word and
every nonselected word in a valid lane to positive infinity. This prevents
retained stale cells from escaping through the public output while preserving
selected-only device initialization and reset accounting.

For every normal case, compare each valid lane with an independent bounded
Dijkstra query on that lane's selected induced subgraph. Exact-representable
fixtures require bitwise equality. Width one must agree with standalone
frontier push in final bits and terminal classification. Do not require equal
GPU round counts when legal concurrent scheduling changes the order of strict
atomic improvements.

Enabled and disabled per-lane convergence must produce identical final
selected-region distance bits, reached masks, and miss masks. A lane with no
next-frontier bit must receive no synthetic work in the disabled mode. Normal
convergence must set every valid lane converged; maximum-round and error exits
must publish neither reached nor miss.

The grid-stride initialization must reset every configured lane's convergence
record, including padded lanes and retained capacity reused after a wider run.
A record may be published only by a clean accepted continue, convergence, or
maximum-round transition. Error, invalid-controller, and terminal no-op
transitions publish no convergence proof. Include a mixed clean maximum-round
case in which only lanes with a proven empty next frontier receive a record.

## 4. Validate queue and activity-mask invariants

The following are required device round-boundary invariants. The current
public HIP output and `bfnew.batched_frontier_push_gpu` CTest do **not** expose
a complete per-round queue-to-mask boundary ledger. They exercise controller,
mask-admission, capacity, counter-identity, and no-out-of-bounds hot-path
guards, but passing that CTest alone is not proof of the full list below. Before the
combined campaign claims device ledger correctness, add a future API or a
temporary/manual Debug diagnostic that snapshots the semantic queue prefixes,
corresponding mask images, controller masks, and error state after every
accepted boundary. Retain that diagnostic evidence for:

1. Every current queue entry is a unique, in-range selected vertex.
2. Every queued vertex has a nonzero activity mask admitted by its owner tile.
3. Every nonzero current activity mask has exactly one current queue entry.
4. The OR of current per-vertex masks equals the controller execute mask.
5. The next queue is empty before a round begins.
6. A destination is appended only by the atomic OR that observes an old next
   mask of zero.
7. Later successful updates may add lane bits or improve labels without
   appending a duplicate vertex.
8. The OR of complete next masks equals `next_frontier_lane_mask`.
9. A consumed current mask is zero before its slot is recycled.
10. A queue claim at or beyond capacity sets `queue_overflow` and performs no
    out-of-bounds write.

The default queue capacity is one vertex entry per graph vertex and cannot
overflow when those invariants hold. A smaller explicit capacity is a
validation seam, not a silent truncation or production memory policy.

For shared-source initialization, verify that one distinct source vertex
creates one physical queue entry whose mask contains every source lane, while
each source lane has its own positive-zero distance word. No query source set
may be merged into another.

## 5. Validate controls and persistent residency

For ordinary controls, retain the actual engine-round dispatches, controller-
advance dispatches, controller copies, host checks, synchronizations, and the
reported `ordinary_grid_blocks` value.
Per-round polling observes after one ordered pair. Chunked polling must submit
the complete configured K pairs before observation; pairs already queued after
device `done` must leave controller state, round count, and queue parity
unchanged.

Persistent control must perform initialization, all queue rounds and swaps,
controller transitions, and terminal status in one cooperative launch. It must
perform no host frontier-size or convergence polling. Every workgroup must
reach every cooperative-grid barrier, including barriers surrounding the
terminal transition.

Query legal residency from the actual selected persistent and ordinary
batched-frontier kernels using
`hipOccupancyMaxActiveBlocksPerMultiprocessor`. Retain the block size, dynamic
shared memory, registers per thread, active blocks per WGP, WGP count, and
cooperative grid size. Reject a requested fixed grid that exceeds the real
kernel limit. Never hard-code the earlier 160-block probe ceiling.
The reported `ordinary_grid_blocks` must equal the occupancy-derived resident
cap further limited by queue-capacity work blocks; it is launch metadata and
must not be reported as measured occupancy.

## 6. Check workspace capacity and selected initialization

For graph vertex count `V`, lane width `W`, and queue capacity `Q`, verify the
checked scratch formula:

```text
distance bytes       = 4 * V * W
two queue bytes      = 8 * Q
two mask bytes       = 8 * V
total scratch bytes  = 4 * V * W + 8 * Q + 8 * V
```

The Phase 13 generic one-slot estimate included only the distance component.
It excluded queues and activity masks and must not be cited as the complete
Phase 16 allocation. Record resident graph bytes, all retained workspace
components, allocator overhead if available, runtime free memory before and
after allocation, and the maximum safe concurrent workspace count.

Verify selected-only initialization separately from retained capacity:

- one distance word for each admitted vertex/lane;
- both activity-mask words for every union vertex;
- source seeds as the zero-valued subset of selected distance writes;
- one initial queue/mask write per distinct source vertex;
- a grid-stride zero reset for every configured lane's convergence record; and
- no semantic read from padded or nonselected distance cells.

The real full-vertex/retained-CSR choice remains provisional until these
measurements are available on real plans. Portable descriptor equivalence is
not a compact HIP implementation. The bounded portable parity gate compares
retained masks with compact descriptors across every vertex/lane output bit,
terminal status mask, completed-round result, and convergence-round record.

## 7. Collect algorithmic work and sharing evidence

Use Light or Debug instrumentation only for counter runs, never for final
timing. `Light` exposes aggregate sharing, work, and queue-high-water evidence.
Retain at least:

- physical frontier vertex entries and active vertex/lane pairs;
- multi-lane frontier vertices and average active lanes per worklist vertex;
- CSR runs considered, visited, and skipped, plus active lanes over visited
  runs;
- logical shared edge-record requests and lane-edge relaxations;
- multi-lane shared edge records;
- successful strict updates and queue high-water;
- active/valid/configured-width/wave32 lane capacity and padding;
- each lane's first empty-next-frontier round and later batch tail; and
- selected distance/mask reset and source-seed bytes.

Run `Debug` separately when retaining the finer queue/atomic vocabulary:

- current-mask exchanges, next-mask ORs, and controller-mask ORs;
- distance atomic attempts and successful strict updates;
- unique next vertex/lane activations and physical queue claims;
- queue entries saved by lane merging;
- same-lane and total duplicate suppressions; and
- overflow events.

The exact merge identity is:

```text
queue entries saved by lane merging
  = unique next vertex/lane activations - physical queue claims
```

Logical shared edge loads and requested bytes do not prove a physical cache
load, L2 transaction, or memory bandwidth. Arithmetic lane utilization is not
measured GPU occupancy. Because disabled convergence still forbids invented
frontier work, semantic work avoided solely by enabling per-lane convergence
may legitimately be zero; report that result rather than manufacturing work.

## 8. Timing comparison with the other batched engines

Timing begins only after correctness passes. Use instrumentation None, keep the
resident graph and workspace capacities stable, warm up the device, and use
status-only result paths. Separate batch preparation, SSSP device-timeline
span, result transfer, and end-to-end host wall time. In host-poll modes the
event span may include host observation/re-enqueue idle gaps and is not summed
kernel-active time.

Compare batched frontier, batched Jacobi, and batched dense only on identical
graph/query/plan fingerprints, lane widths, valid masks, controls, convergence
settings, block/grid policy, and execution order. Interleave configurations to
reduce thermal/order bias. Report at least P50/P95/P99 wall and event spans,
per-query latency, total throughput, and long-tail batches.

The bounded correctness CTest is not a throughput harness. If no reviewed
combined-campaign driver exists, add one in its own later task before measuring;
do not invent command-line arguments or treat repeated CTest execution as a
representative workload.

## 9. Trace and profiler runs

Profiler runs use separate processes from final timing. Retain tool and counter
availability first:

```bash
mkdir -p build/phase16-hip/profiler
rocprofv3 --version \
  >build/phase16-hip/profiler/rocprofv3-version.txt 2>&1
rocprofv3-avail -d 0 list --pmc \
  >build/phase16-hip/profiler/pmc-list.txt 2>&1
rocprofv3-avail -d 0 info --pmc \
  >build/phase16-hip/profiler/pmc-info.txt 2>&1
```

Use `rocprofv3-avail -d 0 pmc-check ...` before collecting each candidate
group. Use the trace and compatible PMC procedure in
`docs/PHASE12_GPU_VALIDATION.md`. Retain kernel and memory-copy traces,
registers, waves/occupancy, physical L2 reads/writes, memory-unit activity, and
compatible atomic/write-stall evidence. Split incompatible counter groups into
separate passes. Unavailable counters remain unavailable; never encode them as
zero. Profiler-instrumented runtimes are not ordinary performance samples.

## 10. Required final report

The Phase 16 device report must answer with direct evidence references:

1. Are every width, lane, control/K, and convergence setting correct without
   cross-lane contamination?
2. Do queue/mask invariants and explicit overflow pass on device, with a
   retained full boundary ledger from the required diagnostic rather than the
   current CTest alone?
3. How many vertices and edge records are shared across lanes?
4. How many physical queue entries are saved by lane merging, and at what mask
   atomic cost?
5. What are distance atomic attempt/success, duplicate suppression, run-skip,
   queue high-water, and lane-utilization distributions?
6. How much semantic work is actually avoided by per-lane convergence, and how
   many tail rounds remain after each lane finishes?
7. What are complete retained workspace capacity and selected reset traffic
   after adding both queues and both activity masks?
8. What registers, legal residency, actual occupancy, and physical L2/memory
   behavior occur on the target?
9. Which width/control/convergence setting maximizes end-to-end query
   throughput?
10. How does batched frontier throughput compare with correctness-matched
    batched Jacobi and dense push on the same plans?

Every conclusion names graph/query/plan fingerprints, source revision,
hardware/software identity, configuration, instrumentation level, timing stage,
and profiler evidence. CPU timings, logical work counts, and source inspection
answer no device-performance question.

## Current conclusion

All Phase 16 HIP compilation, device correctness, occupancy, physical memory,
latency, throughput, and cross-engine comparison questions are pending. The
full-vertex/retained-CSR choice remains provisional. No batch width, control,
convergence setting, frontier batching benefit, or production configuration is
recommended. Edge balancing, wave aggregation, and adaptive frontier
scheduling remain unimplemented future work. Phase 17 expansion/replanning is
now implemented under the CPU-only acceptance policy; its separate deferred
procedure is `docs/PHASE17_GPU_VALIDATION.md`. This campaign stays deferred
until all implementation phases are complete.
