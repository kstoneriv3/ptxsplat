# Benchmark And Roofline Protocol

The checked-in CLIs produce machine-readable JSON and must be run through the
project container:

```bash
./scripts/docker-run.sh -- python3 -m benchmarks.garden --help
./scripts/docker-run.sh -- python3 -m benchmarks.roofline --help
```

The smoke defaults are intentionally small: garden uses a 320x180 render with
three warmups and ten measured iterations; the roofline probe uses 64 MiB copy
buffers and a 2048-square GEMM. A local smoke record can be generated with:

```bash
./scripts/docker-run.sh -- python3 -m benchmarks.garden \
  --output benchmark-results/garden-smoke.json
./scripts/docker-run.sh -- python3 -m benchmarks.roofline \
  --output benchmark-results/roofline-smoke.json
```

## Primary workload

Use `assets/test_garden.npz` with one pinhole camera, packed classic
rasterization, SH degree 3, a background, and an MSE backward pass. Measure
`scene_grid` 1, 3, and 7 at 720p and 1080p. Secondary sweeps cover precomputed
RGB, overlap density, Gaussian count, and 4K output.

The publication matrix is exposed as a stable preset:

```bash
./scripts/docker-run.sh -- python3 -m benchmarks.garden --full \
  --output benchmark-results/garden-full.json
```

`--full` expands to scene grids 1/3/7, 720p/1080p, SH degree 3, black
background, forward and combined forward+backward, 20 warmups, 100 measured
iterations, and five rounds. It deliberately overrides the corresponding CLI
arguments. Use explicit lists for feature sweeps, for example:

```bash
./scripts/docker-run.sh -- python3 -m benchmarks.garden \
  --scene-grids 1,3 --resolutions 360p,720p \
  --color-modes rgb,sh0,sh1,sh2,sh3 \
  --backgrounds 'none;black;white;0.1,0.2,0.3' \
  --workloads forward,forward-backward \
  --warmup 20 --iterations 100 --rounds 5 \
  --output benchmark-results/garden-feature-sweep.json
```

## Timing

- Verify the GPU is idle and warm it to a stable clock state.
- For publication runs, run 20 untimed warmups, 100 CUDA-event-timed
  iterations, and five rounds.
- Report median, interquartile range, and a bootstrap 95% confidence interval.
- Report forward, combined forward+backward, and peak allocated memory. Measure
  isolated backward and individual stages separately when optimizing a stage.
- Record git commit, container image reference (plus its immutable ID when
  available), driver, CUDA, PyTorch, GPU clocks, temperature, power, input
  counts, visible Gaussians, intersections, and pixel-Gaussian evaluations.

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

`benchmarks.roofline` is a lightweight first-pass probe, not a replacement for
Nsight Compute. It records host enqueue and serialized GPU time for a scalar
kernel, device-copy bandwidth with read and write bytes counted, and IEEE FP32
GEMM throughput with TF32 disabled. Increase buffer and GEMM sizes for a stable
device ceiling:

```bash
./scripts/docker-run.sh -- python3 -m benchmarks.roofline \
  --dram-bytes 268435456 --gemm-size 8192 \
  --warmup 20 --iterations 100 \
  --output benchmark-results/roofline-full.json
```

## Experiment records

Every candidate emits JSON containing the hypothesis, source/backend, build
flags, correctness result, timings, counters, register/shared-memory use,
spill counts, and disposition. Failed experiments remain in the local results
log so the agent does not repeat them; large `.ncu-rep`, cubin, and SASS files
remain untracked.

The benchmark JSON records raw timing samples, summary statistics,
deterministic seed, exact arguments, git state, Python/PyTorch/CUDA versions,
ptxsplat/gsplat versions, driver, GPU identity, and a point-in-time
clock/temperature/power sample. Garden records total/visible Gaussian and
intersection counts from an untimed finite-output preflight. Add
optimization-specific hypotheses, compiler data, correctness results, and
disposition when promoting these raw measurements into experiment records.
