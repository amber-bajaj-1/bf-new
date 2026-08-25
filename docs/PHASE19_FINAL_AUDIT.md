# Phase 19 Final Audit and Evidence-Based Recommendation

## Final outcome

Phase 19 adds no shortest-path, batching, expansion, reconstruction, congestion,
or adaptive-selection algorithm. It closes the implementation campaign with a
fail-closed audit of the features and evidence produced through Phase 18.

The local audit result is **insufficient representative evidence; no production
configuration is recommended**. In particular, Phase 19 does not choose a
configuration by enum order, bounded-fixture timing, analytical memory size,
source inspection, or intuition. Every engine, control, chunk size, persistent
grid policy, batch width, planner threshold, and expansion schedule remains an
explicit configurable choice.

The configured FPGA24 data root contains both generation inputs:

```text
logicnets_jscl_unrouted.phys
logicnets_jscl.netlist
```

Those files are not solver inputs. The deterministic Phase 7 workload builder
must combine them with the device resources to emit a deeply validated,
versioned query artifact. That required artifact,
`out/phase12-workload/logicnets_jscl.padding1.v1.bfqueries`, is absent. The
partial Phase 7 run did emit and round-trip
`out/phase7/xcvu3p.v1.bfgraph`, but it was stopped before the real query
artifacts and complete 13-input scan were produced.

No HIP compiler, compatible GPU execution, accepted profiler trace/PMC set,
representative corpus run, matched CPU baseline, or normalized Phase 18
per-query tail provenance is present in the local evidence. These values are
unavailable, not measured zero.

## Evidence classes

The audit uses exactly four mutually exclusive classes:

1. **Implemented and correctness-tested** — executable portable behavior or an
   evidence contract has bounded Release and sanitizer correctness coverage.
2. **Implemented and performance-profiled** — representative correctness,
   ordinary timing, and compatible target-profiler evidence exist for the same
   workload/configuration identity.
3. **Implemented but not yet representative** — an implementation exists, but
   it lacks representative device/corpus evidence.
4. **Designed but deferred** — the contract or measurement question exists,
   but no qualifying implementation result has been accepted.

The fixed machine report has 56 feature rows. Its canonical local partition is
36 class-1, zero class-2, 12 class-3, and 8 class-4 rows. Source-level HIP code,
fake-HIP syntax, portable work counters, bounded host timings, analytical byte
counts, and profiler schemas do not satisfy class 2.

| Audited feature | Class | Local evidence boundary |
| --- | ---: | --- |
| Portable standalone Jacobi min-plus CSC pull | 1 | Bounded Dijkstra agreement, all three controls, and all five chunk sizes |
| Portable standalone dense chaotic CSR push | 1 | Bounded Dijkstra agreement, controller semantics, and atomic-domain invariants |
| Portable standalone active-frontier CSR push | 1 | Bounded Dijkstra agreement, queue/deduplication invariants, and overflow behavior |
| Standalone persistent, chunked, and per-round control contracts | 1 | Portable semantic, accounting, and fail-closed tests |
| Controlled shootout manifest, catalog, evidence gates, summaries, and serialization | 1 | Deterministic bounded tests only; no representative samples |
| Deterministic overlap planner and both run representations | 1 | Width 32/16/8 plans, exact masks, padding, and permutation invariance |
| Checked workspace and memory-accounting models | 1 | Bounded/analytical model evidence only |
| Portable batched Jacobi pull at widths 1/8/16/32 | 1 | All controls, convergence settings, padding, and bounded oracle agreement |
| Portable batched dense chaotic push at widths 1/8/16/32 | 1 | All controls, representations, padding, and bounded oracle agreement |
| Portable batched active-frontier push at widths 1/8/16/32 | 1 | All controls, queue/mask invariants, padding, and bounded oracle agreement |
| Portable all-query expansion/replanning across all three engines | 1 | All four schedules, retries, fallback/failure, identity, and deterministic traces |
| Portable compact reconstruction and no-congestion result boundary | 1 | Stable-edge backtracking, exact validation, byte/stage accounting, and deterministic quality sampling |
| HIP standalone engines and retained device foundation | 3 | Implemented source; no HIP compilation or device execution locally |
| HIP batched engines at widths 1/8/16/32 | 3 | Implemented source; no device correctness or representative comparison |
| HIP expansion adapter and compact reconstruction | 3 | Implemented source; no device/corpus run or transfer trace |
| HIP+FPGAIF artifact campaign driver | 3 | Implemented source; required query artifact and HIP environment absent |
| Full-vertex/retained-run-mask workspace selection | 3 | Provisional bounded/model decision, not a production memory result |
| Representative latency, throughput, work, memory, copy, and tail evidence | 4 | Campaign designed; qualifying evidence absent |
| Profiler-supported bottleneck conclusions and configuration recommendation | 4 | Trace and PMC evidence absent |
| Adaptive-hybrid experiment decision | 4 | Decision deferred until standalone results are representative |

Host correctness never promotes the corresponding HIP row to device
correctness. Likewise, class 3 does not imply a speedup or a safe production
default.

## Required surface audit

The implementation inventory contains all three standalone engines—Jacobi
pull, dense chaotic push, and active-frontier push—and all three independent
batched counterparts. Their portable tests cover persistent cooperative,
chunked host poll, and per-round host poll controls. Chunked controls cover
`K = 2, 4, 8, 16, 32`. Batched engines cover widths `1, 8, 16, 32`; the
standard overlap family is `32, 16, 8`, while width 1 is the scalar baseline.

Phase 17 covers one-ring, fixed-larger-ring, doubled-margin, and hybrid
small-first/doubling expansion schedules. Phase 18 covers compact target/path
production, terminal-generation ownership, exact compact validation, transfer
accounting, cold/pre-grown-warm stage boundaries, and a deterministic bounded
versus unbounded quality sample.

The exact HIP-off executable/CTest evidence pairs for the final campaign
surface are:

| Surface | Executable | CTest |
| --- | --- | --- |
| Standalone Jacobi | `bfnew_jacobi_pull_test` | `bfnew.jacobi_pull` |
| Standalone dense chaotic | `bfnew_dense_chaotic_push_test` | `bfnew.dense_chaotic_push` |
| Standalone frontier | `bfnew_frontier_push_test` | `bfnew.frontier_push` |
| Shootout evidence layer | `bfnew_shootout_test` | `bfnew.shootout` |
| Overlap planner | `bfnew_batch_plan_test` | `bfnew.batch_plan` |
| Workspace model | `bfnew_batch_workspace_test` | `bfnew.batch_workspace` |
| Batched Jacobi | `bfnew_batched_jacobi_pull_test` | `bfnew.batched_jacobi_pull` |
| Batched dense chaotic | `bfnew_batched_dense_chaotic_push_test` | `bfnew.batched_dense_chaotic_push` |
| Batched frontier | `bfnew_batched_frontier_push_test` | `bfnew.batched_frontier_push` |
| Expansion/replanning | `bfnew_batched_expansion_test` | `bfnew.batched_expansion` |
| Compact no-congestion results | `bfnew_compact_paths_test` | `bfnew.compact_paths` |
| Final audit | `bfnew_final_audit_test` | `bfnew.final_audit` |

Optional HIP CTests remain deferred. Their exact executable/CTest pairs are:

| Surface | Executable | CTest |
| --- | --- | --- |
| Resident transfer | `bfnew_device_transfer_test` | `bfnew.device_transfer` |
| Standalone Jacobi | `bfnew_jacobi_test` | `bfnew.jacobi` |
| Standalone dense chaotic | `bfnew_dense_chaotic_push_gpu_test` | `bfnew.dense_chaotic_push_gpu` |
| Standalone frontier | `bfnew_frontier_push_gpu_test` | `bfnew.frontier_push_gpu` |
| Batched Jacobi | `bfnew_batched_jacobi_hip_test` | `bfnew.batched_jacobi` |
| Batched dense chaotic | `bfnew_batched_dense_chaotic_push_hip_test` | `bfnew.batched_dense_chaotic_push_gpu` |
| Batched frontier | `bfnew_batched_frontier_push_hip_test` | `bfnew.batched_frontier_push_gpu` |
| Expansion/replanning | `bfnew_batched_expansion_hip_test` | `bfnew.batched_expansion_gpu` |
| Compact paths | `bfnew_compact_path_results_test` | `bfnew.compact_path_results` |

The campaign tools are `bfnew_gpu_shootout`, `bfnew_shootout_report`,
`bfnew_shootout_profile_import`, `bfnew_gpu_batched_expansion`,
`bfnew_build_fpga_workload`, `bfnew_gpu_probe`,
`bfnew_gpu_barrier_benchmark`, and `tools/run_phase6_profiler.sh`. Phase 19
adds no new GPU runner.

## Final comparison

An unavailable result has no numeric value, provenance reference, or selected
configuration. The local final comparison is therefore:

| Required comparison | Phase 19 result | Missing qualifying evidence |
| --- | --- | --- |
| Best engine | Unavailable | Complete correctness-gated, matched standalone and batched candidate matrix |
| Best Jacobi control mode | Unavailable | Representative matched device timing/profile evidence |
| Best dense-chaotic control mode | Unavailable | Representative matched device timing/profile evidence |
| Best frontier control mode | Unavailable | Representative matched device timing/profile evidence |
| Best Jacobi ordinary-kernel `K` | Unavailable | Complete comparable `K=2/4/8/16/32` device matrix |
| Best dense-chaotic ordinary-kernel `K` | Unavailable | Complete comparable `K=2/4/8/16/32` device matrix |
| Best frontier ordinary-kernel `K` | Unavailable | Complete comparable `K=2/4/8/16/32` device matrix |
| Best Jacobi persistent grid policy | Unavailable | Actual-kernel occupancy plus matched timing/profile evidence |
| Best dense-chaotic persistent grid policy | Unavailable | Actual-kernel occupancy plus matched timing/profile evidence |
| Best frontier persistent grid policy | Unavailable | Actual-kernel occupancy plus matched timing/profile evidence |
| Best batching width | Unavailable | Representative width 1/8/16/32 correctness-gated comparison |
| Best overlap-planner thresholds | Unavailable | Representative threshold sweep and comparable batch identities |
| Best expansion schedule | Unavailable | Complete comparable four-schedule all-query evidence |
| P50/P95/P99 latency | Unavailable | Ordinary warmed representative timing samples |
| All-query throughput | Unavailable | Complete representative artifact execution |
| Cold and pre-grown warm end-to-end time | Unavailable | Measured complete Phase 18 stage ledgers |
| Box-miss, expansion, and fallback rates | Unavailable | Complete representative terminal ledger |
| Path-quality sample | Unavailable | Frozen representative sample and matched unbounded Dijkstra results |
| Resident, workspace, and compact-result memory footprint | Model only; representative value unavailable | Actual allocation/peak evidence on the target and corpus |
| Synchronization and copy counts | Portable semantics only; representative physical counts unavailable | Device traces and matched controller/copy ledgers |
| Profiler-supported bottleneck conclusions | Unavailable | Accepted, normalized trace and compatible PMC evidence |

The overlap thresholds are configurable in the planner API, but the current
artifact driver records the default `1/8` minimum Jaccard and `2/1` maximum
union inflation without exposing a threshold sweep. Current memory models are
checked, but the Phase 18 report does not make a representative physical
resident/workspace peak claim. These limitations reinforce the unavailable
result; they are not filled from the bounded fixtures.

## Profiler audit

The typed profiler inventory has ten canonical rows. The trace row requires
normalized GPU-active time from kernels strictly inside the recorded begin/end
markers. The nine PMC rows require all of these semantic fields, subject to the
target's compatible counter groups:

- L2 hit percentage;
- L2 read bytes;
- L2 write bytes;
- occupancy percentage;
- memory-unit busy percentage;
- waves;
- vector instructions;
- scalar instructions; and
- memory instructions.

`tools/run_phase6_profiler.sh` begins with utilization,
instruction/wave, and L2/memory candidate groups and recursively splits
incompatible groups. No locally accepted group, marker-delimited SSSP trace,
nonempty compatible PMC result, or normalized import exists. All profiler
fields and all profiler-supported bottleneck conclusions are therefore
unavailable.

A future measured profiler row must use the same configuration fingerprint as
the profiler-bottleneck comparison, and that comparison must match the
best-engine selection. Percentage rows are bounded to 0 through 100; nonfinite,
negative, and signed-negative-zero values fail canonical validation.

## Explicit Phase 19 questions

These are the seven Phase 19 questions, separate from the seven earlier Phase
12 shootout questions. Each local answer is `insufficient_evidence`:

1. **Does Jacobi min-plus pull benefit from the GPU?** Insufficient evidence;
   no matched CPU/device baseline exists.
2. **Does eliminating distance atomics outweigh dense incoming-edge scans?**
   Insufficient evidence; no representative timing/traffic comparison exists.
3. **Does chaotic propagation reduce enough rounds to justify its atomics?**
   Insufficient evidence; no matched round, atomic, timing, and profiler data
   exists.
4. **Does frontier work reduction offset queue and deduplication overhead?**
   Insufficient evidence; no representative queue/work/latency evidence exists.
5. **Does cooperative persistence beat chunked execution?** Insufficient
   evidence; no complete correctness-gated control comparison exists.
6. **Is overlapping batching necessary for adequate utilization?**
   Insufficient evidence; widths 1/8/16/32 have not been profiled on the target.
7. **Which tail-query class dominates all-query runtime?** Insufficient
   evidence; no normalized per-query tail attribution exists.

## Recommendation and stop boundary

The Phase 19 production recommendation is absent. It carries no configuration
IDs and no evidence references. All runtime toggles remain configurable. The
audit also does not recommend for or against an adaptive hybrid experiment:
that decision is deferred until representative standalone results are clear.
No adaptive hybrid is implemented.

In a later measured report, each question's supporting configuration
fingerprints must be nonzero, strictly increasing, and unique within that
answer, and its evidence fingerprint must bind them to the same evidence
snapshot. The audit stores only a normalized candidate-catalog fingerprint and
a complete-matrix attestation supplied by a trusted normalized campaign. It
enforces internal identity and canonicality, but does not ingest raw timing
rows, reconstruct catalog membership, or independently rerank candidates.
A legal frontier recommendation carries a nonzero queue capacity; a
non-frontier recommendation carries zero so no irrelevant queue toggle enters
its identity.

Congestion costs, resource ownership/conflicts, occupancy or historical-cost
updates, disjoint batching, an exit lower-bound certificate, and any Phase 20
work are outside this campaign. Phase 19 is the terminal implementation phase.

## Missing gates

The canonical local report carries these 15 blockers in this exact order:

1. missing representative query artifact;
2. missing HIP compiler validation;
3. missing GPU device validation;
4. missing comparable CPU baseline;
5. missing representative correctness;
6. missing representative timing;
7. missing dependency trace;
8. missing compatible PMC evidence;
9. incomplete candidate matrix;
10. missing all-query evidence;
11. missing normalized tail-attribution evidence;
12. no unique winner;
13. missing memory evidence;
14. missing synchronization/copy evidence; and
15. missing path-quality evidence.

A later evidence review may change an unavailable comparison only after all
applicable gates are satisfied:

1. generate and deeply validate the versioned `logicnets_jscl` `.bfqueries`
   artifact from the existing `.phys`, `.netlist`, and device resources;
2. compile the optional sources with the real HIP compiler and pass every
   bounded device correctness CTest;
3. run a fingerprinted representative corpus with complete comparable
   configuration rectangles and ordinary warmed timing;
4. collect accepted marker-delimited traces and compatible PMC groups;
5. retain a matched CPU baseline for GPU-benefit claims;
6. normalize all profiler units/provenance without converting missing values to
   zero; and
7. add Phase 18 configuration/workload provenance and per-query tail
   attribution sufficient to join every conclusion to its evidence.

The detailed deferred procedures remain in
`docs/PHASE12_GPU_VALIDATION.md`, `docs/PHASE17_GPU_VALIDATION.md`, and
`docs/PHASE18_GPU_VALIDATION.md`. Those runbooks do not become local Phase 19
evidence merely because their commands exist.

## Local validation

The default final gate remains HIP-off and dependency-minimal:

```bash
cmake -S . -B build/phase19-cpu \
  -DBFNEW_ENABLE_HIP=OFF \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase19-cpu --parallel
ctest --test-dir build/phase19-cpu --output-on-failure
ctest --test-dir build/phase19-cpu \
  -R '^bfnew[.]final_audit$' --output-on-failure
```

Sanitizer coverage uses a separate tree:

```bash
cmake -S . -B build/phase19-asan \
  -DBFNEW_ENABLE_HIP=OFF \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build/phase19-asan --parallel
ctest --test-dir build/phase19-asan --output-on-failure
```

Final measured local evidence from the clean integrated run:

```text
HIP-off Release: 20/20 passed in 2.12 seconds
  bfnew.final_audit: 1/1 passed in 0.01 seconds
HIP-off ASan+UBSan: 20/20 passed in 3.63 seconds
  bfnew.final_audit: 1/1 passed in 0.04 seconds
Strict Phase 19 source/public-header/test syntax: passed warning-clean with
  C++20, -Wall, -Wextra, -Wpedantic, -Wconversion, -Wsign-conversion,
  -Wshadow, and -Werror
```

These local checks validate the audit contract and its refusal to overclaim.
They are not HIP compilation, GPU correctness, representative timing, or
profiler evidence.
