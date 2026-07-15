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

## Component-kernel promotion

Set an operation-specific isolated-gain threshold before timing. The primary
performance gate is that the paired isolated median and its bootstrap 95% lower
confidence bound both clear that threshold. Do not require every component
kernel to produce a fixed 2% whole-pipeline gain.

Also require all of the following:

- the paired whole-pipeline gain has a bootstrap 95% lower bound above zero;
- with baseline stage share `f = stage_ms / pipeline_ms`, the pipeline median
  gain is no greater than the isolated gain and is between `0.5 * f * gain` and
  `2.0 * f * gain`; record the inputs, prediction, and ratio;
- exact/tolerance-preserving correctness passes, and no required publication
  case regresses by more than 2% at the median;
- the promoted invocation has no stack, local memory, or spills, and does not
  lose theoretical residency or achieved occupancy;
- no adversarial traffic case grows total L2 sectors by more than 5%.

## Resource ceilings

Measure launch latency, DRAM and L2 bandwidth, coalesced and gather loads, FP32
and special-function throughput, atomic accumulation, scan, and key/value sort.
Use dynamic operation and byte counts with Nsight Compute to estimate a
sustained lower bound for each stage. The end-to-end roof is the sum of stage
lower bounds, not a peak-FLOP marketing number.

`benchmarks.roofline` is a lightweight first-pass probe, not a replacement for
Nsight Compute. It records host enqueue and serialized GPU time for a scalar
kernel, device-copy bandwidth with read and write bytes counted, and IEEE FP32
GEMM throughput with TF32 disabled. Each DRAM and GEMM timing sample contains a
batch of operations (ten copies or five GEMMs by default) to sustain work between
event synchronizations, and all raw per-operation samples are retained. Increase
buffer and GEMM sizes for a stable device ceiling:

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

## Promoted raster kernel ceiling

`benchmarks.kernel_ceiling` constructs an operation-aware ceiling for the
promoted SM120 RGB raster forward and backward kernels. The publication run
requires Nsight Compute privileges and must use the profile container:

```bash
./scripts/docker-run.sh --profile -- \
  python3 -m benchmarks.kernel_ceiling run \
  --output-dir benchmark-results/kernel-ceiling \
  --thermal-warmup-seconds 8 \
  --rounds 5 --samples-per-round 20 \
  --repetitions-per-sample 5 --raster-samples 100 \
  --ncu-raster-launches 5
```

The workload is the primary grid-7 garden case at 1920x1080, packed RGB,
black background, tile size 16, and `absgrad=False`. The command times focused
native CUDA probes, captures target-function SASS, and profiles dynamic opcode
and traffic counts for the exact promoted kernels. Probe kernels use 256-thread
CTAs and dynamic shared memory to reproduce the relevant five- or seven-CTA
residency. The REDG probe uses four- and eight-warp intra-CTA contention plus an
eight-CTA stress case. Balanced and skewed barriers, vector shared loads and
stores, independent and dependent MUFU instructions, and launch/grid tail are
measured separately. Raw event samples and round-bracketing clock,
temperature, and power telemetry are retained. Probe rates use only rounds
starting at least 90% of the maximum observed round-start SM clock and at or
below 85 C; rejected transition or throttled rounds remain in the raw record.

For the promoted backward invocation, the exact minimum is nine scalar warp
sums and nine lane-zero REDG FP32 atomics for every warp-active Gaussian. Each
sum has five shuffle/add stages. Keep all nine mandatory sums, including
opacity, adjacent before optional absgrad work so nvcc can emit each XOR stage
across the independent values. The initial per-warp maximum compiles to one
`REDUX` instruction. The analysis requires the dynamic identities
`SHFL = active_events * 9 * 5`, `REDG = active_events * 9`, and
`REDUX = launched_warps` to hold. It also maps per-PC counts onto captured SASS
to distinguish `MUFU.EX2` from `MUFU.RCP`, barrier variants, and vector shared
instructions. The reduction and REDG probes execute the same nine-value
`absgrad=False` path, and a separate dependent `REDUX` probe covers the initial
warp maximum. Probe output checksums and probe NCU counts guard against compiler
elimination.

NCU CSV time and byte metrics are normalized before analysis while retaining
their original value and unit. Time units `ns`, `us`, `ms`, and `s` normalize to
milliseconds. NCU decimal `byte`/`Kbyte`/`Mbyte`/`Gbyte` units normalize to
bytes; binary `KiB`/`MiB`/`GiB` spellings use powers of 1024. Per-block byte
units retain the `/block` qualifier. An unknown unit aborts analysis rather than
falling through to an unscaled number.

Within one kernel, the operation-aware empirical resource target is the maximum
of FP32, DRAM, L2, shared-memory, MUFU, CTA barrier, reduction
issue/dependency, REDG atomic, and launch/grid-tail terms. Do not sum terms that
can overlap. EX2 and RCP demands are summed only within their shared MUFU path,
and LDS and STS demands are summed only within their shared-memory path. Forward
and backward targets are added because those kernels execute sequentially. A
candidate resource rate may not be lower than throughput already sustained by
the exact kernel.

These operation-aware terms divide exact work counts by independently measured
probe rates. The resulting maximum is an empirical sustained resource target,
not a physical latency lower bound and not evidence that a direct-PTX kernel can
attain all rates simultaneously. The historical FP32/DRAM/L2-only roofline is
reported separately as a physical lower-bound model.

Primary current efficiency and residual use 100 non-profiled CUDA-event wrapper
samples. Forward contains one promoted kernel launch. Backward also performs
four required output zero-initializations, so its wrapper latency is a
conservative upper bound on kernel latency and produces a lower-bound
efficiency. NCU replay duration is retained only for dynamic-count validation,
deriving its L2 peak, and a profile-comparable bridge to the historical 18.48%
metric; it is not substituted for event latency.

The terminal `benchmark-results/kernel-ceiling/analysis.json` distinguishes:

- the historical optimistic FP32/DRAM/L2-only roofline;
- the operation-aware empirical sustained resource target;
- profile-comparable NCU replay latency, which is not the primary latency;
- algorithm-changing headroom, without asserting an achievable direct-PTX
  bound.

Sensitivity uses the maximum observed four-warp REDG throughput and minimum
launch latency for the optimistic case, medians with representative eight-warp
contention for the central case, and slow samples plus skew and cross-CTA
contention for the conservative case. The 25%-of-ceiling criterion
passes only when the lowest empirical target divided by the highest measured
kernel latency is at least 25%. The less-than-10% residual criterion passes
only when `current / empirical_resource_target - 1` is below 10% at the worst
end of the range. Passing either comparison would still characterize the model;
it would not prove that a direct-PTX implementation can attain the target. A
failed boolean is a result, not permission to alter the model or inputs.
