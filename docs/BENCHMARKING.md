# Benchmark and roofline protocol

## Primary workload

Use `assets/test_garden.npz` with one pinhole camera, packed classic
rasterization, SH degree 3, a background, and an MSE backward pass. Measure
`scene_grid` 1, 3, and 7 at 720p and 1080p. Secondary sweeps cover precomputed
RGB, overlap density, Gaussian count, and 4K output.

## Timing

- Verify the GPU is idle and warm it to a stable clock state.
- Run 20 untimed warmups, 100 CUDA-event-timed iterations, and five rounds.
- Report median, interquartile range, and a bootstrap 95% confidence interval.
- Report forward, backward, combined training, peak memory, and stage timings.
- Record git commit, container image ID, driver, CUDA, PyTorch, GPU clocks,
  temperature, power, input counts, visible Gaussians, intersections, and
  pixel-Gaussian evaluations.

## Stage measurements

Measure projection, intersection expansion, sorting, offset encoding, SH,
raster forward, and raster backward independently using captured realistic
intermediates. Include eager launch overhead in end-to-end results and report a
CUDA-graph replay measurement separately when useful.

## Resource ceilings

Measure launch latency, DRAM and L2 bandwidth, coalesced and gather loads, FP32
and special-function throughput, atomic accumulation, scan, and key/value sort.
Use dynamic operation and byte counts with Nsight Compute to estimate a
sustained lower bound for each stage. The end-to-end roof is the sum of stage
lower bounds, not a peak-FLOP marketing number.

## Experiment records

Every candidate emits JSON containing the hypothesis, source/backend, build
flags, correctness result, timings, counters, register/shared-memory use,
spill counts, and disposition. Failed experiments remain in the local results
log so the agent does not repeat them; large `.ncu-rep`, cubin, and SASS files
remain untracked.
