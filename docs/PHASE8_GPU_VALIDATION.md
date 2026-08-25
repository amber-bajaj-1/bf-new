# Deferred Phase 8 GPU validation

These commands are intentionally deferred until the maintainer begins the
combined GPU-validation campaign. They are not part of the local Phase 8 pass.
Run them from the `bf-new` project root on the validated AUP `gfx1151` host:

```sh
cmake -S . -B build/phase8-hip \
  -DBFNEW_ENABLE_HIP=ON \
  -DBFNEW_ENABLE_FPGAIF=OFF \
  -DBUILD_TESTING=ON \
  -DBFNEW_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase8-hip \
  --target bfnew_device_transfer_test \
  --parallel
ctest --test-dir build/phase8-hip \
  --output-on-failure \
  -R '^bfnew\.device_transfer$'
```

The configured target and CTest names are
`bfnew_device_transfer_test` and `bfnew.device_transfer`. The test uses only
small synthetic data. It verifies exact device round trips for a hand-checked
layout, the existing Phase 5 spatial/spill fixture, and an empty spatial graph;
then it exercises retained query buffers for a two-source/two-target query and
an engine switch. A tiny test-only kernel validates the raw device CSR/CSC run
bounds, endpoint ownership, maximality, and run-level admission. It is not an
SSSP kernel.

The runtime orders every copy on the supplied stream, and each workspace lease
requires one stream for preparation, result helpers, and retirement. Internal
query staging is HIP page locked. Resident-graph vectors and caller-owned
status/instrumentation outputs may instead be pageable, so do not interpret an
asynchronous API name or trace row as proof of host/device overlap unless the
corresponding host endpoint is page locked.

The runtime helpers reject a different stream, but a kernel launched directly
from a raw `DeviceWorkspaceView` is outside that enforcement boundary. Keep all
such consumers on the lease's preparation stream during this validation.

After the transfer test passes, capture the allocation and copy trace with:

```sh
mkdir -p build/phase8-hip/phase8-trace
rocprofv3 --version \
  >build/phase8-hip/phase8-trace/rocprofv3-version.txt 2>&1
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-directory build/phase8-hip/phase8-trace \
  --output-format csv \
  -- build/phase8-hip/bfnew_device_transfer_test --trace-query-reuse \
  >build/phase8-hip/phase8-trace.log 2>&1
```

Do not infer performance from this trace. Its structural check is that the
resident graph has one allocation/upload sequence and the workspace has one
reservation sequence, while the eight isolated query preparations show no
graph allocation, graph upload, workspace allocation, or second large
engine-scratch allocation. Preserve the CTest output, trace CSV files, and
profiler version with the later GPU-validation evidence.

A bounded-real graph/device round trip is not part of the current small test.
Add and run that bounded sample during the later GPU campaign before claiming
the complete Phase 8 AUP device-evidence checklist. Do not use the full Phase 7
graph for this validation.
