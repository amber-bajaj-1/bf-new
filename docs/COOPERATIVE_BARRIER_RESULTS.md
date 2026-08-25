# Phase 6 Cooperative Barrier Results

## Status

Phase 6 passed under the maintainer's CPU-only acceptance policy. The optional
barrier matrix was not run on `gfx1151`, so this document makes no GPU timing or
occupancy claim. The matrix below is retained as maintainer-run instructions.

## Required matrix

`bfnew_gpu_barrier_benchmark` queries the occupancy of its own kernel for each
legal block size, then evaluates:

- block sizes 128, 256, and 512;
- 1, 2, 4, and maximum legal resident blocks per HIP multiprocessor/WGP; and
- 1, 10, 100, and 1000 cooperative grid barriers.

The raw CSV belongs at
`build-hip/phase6-profile/cooperative-barriers.csv`. If the optional target run
is performed, record the run date, capability JSON, all legal occupancy
ceilings, and a compact summary of the matrix here.

The reported `amortized_kernel_us_per_grid_barrier` divides GPU-event elapsed
time by launches and barrier count. It includes loop and kernel effects and is
not a pure hardware-barrier latency. This isolated microbenchmark does not
predict any of the future SSSP engines' performance.
